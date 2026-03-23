import netCDF4 as nf4
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from metpy.plots import SkewT
import numpy as np
import math as m
from metpy.units import units
import metpy.calc as mpcalc
from metpy.calc import cape_cin, parcel_profile
import matplotlib.patches as mpatches
from metpy.calc import lfc, el
from PIL import Image
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
import re

# This is for demo only

# Ensure the "caesar_skewts" directory exists
output_dir = "caesar_skewts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def natural_sort_key(s):
    # Extracts numbers from the string and converts them to integers for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def file_parser(fp):
    nc = nf4.Dataset(fp)

    def masked_to_filled(array):
        return np.ma.filled(array, fill_value=np.nan)

    # Extracting launch_time and removing "seconds since"
    launch_time_str = nc.variables['launch_time'].units
    launch_time_utc = launch_time_str.split("since", 1)[1].strip()

    data = {'alt': masked_to_filled(nc.variables['alt']), 'altitude_units': nc.variables['alt'].units,
            'tdry': masked_to_filled(nc.variables['tdry']), 'tdry_units': nc.variables['tdry'].units,
            'pres': masked_to_filled(nc.variables['pres']), 'pressure_units': nc.variables['pres'].units,
            'dp': masked_to_filled(nc.variables['dp']), 'dew_point_units': nc.variables['dp'].units,
            'wdir': masked_to_filled(nc.variables['wdir']), 'wdir_units': nc.variables['wdir'].units,
            'wspd': masked_to_filled(nc.variables['wspd']), 'wspd_units': nc.variables['wspd'].units,
            'time': masked_to_filled(nc.variables['time']), 'time_units': nc.variables['time'].units,
            'rh': masked_to_filled(nc.variables['rh']), 'rh_units': nc.variables['rh'].units,
            'u_wind': masked_to_filled(nc.variables['u_wind']), 'u_wind_units': nc.variables['u_wind'].units,
            'v_wind': masked_to_filled(nc.variables['v_wind']), 'v_wind_units': nc.variables['v_wind'].units,
            'theta': masked_to_filled(nc.variables['theta']), 'theta_units': nc.variables['theta'].units,
            'theta_e': masked_to_filled(nc.variables['theta_e']), 'theta_e_units': nc.variables['theta_e'].units,
            'theta_v': masked_to_filled(nc.variables['theta_v']), 'theta_v_units': nc.variables['theta_v'].units,
            'vt': masked_to_filled(nc.variables['vt']), 'vt_units': nc.variables['vt'].units,
            'lat': masked_to_filled(nc.variables['lat']), 'lat_units': nc.variables['lat'].units,
            'lon': masked_to_filled(nc.variables['lon']), 'lon_units': nc.variables['lon'].units,
            'launch_time': launch_time_utc, 'Flight': nc.Flight}

    # Close the file when finish reading
    nc.close()
    return data


def skewT_comparison(data, file_number):
    fig = plt.figure(figsize=(9, 8))
    skew = SkewT(fig)

    p_list = data['pres']
    pres_array = np.array(p_list, dtype=float)

    temp_list = data['tdry']
    temp_array = np.array(temp_list, dtype=float)

    dewp_list = data['dp']
    dewp_array = np.array(dewp_list, dtype=float)

    wind_u_list = data['u_wind']
    wind_u_array = np.array(wind_u_list, dtype=float)

    wind_v_list = data['v_wind']
    wind_v_array = np.array(wind_v_list, dtype=float)

    lat_list = data['lat']
    lat_array = np.array(lat_list, dtype=float)

    lon_list = data['lon']
    lon_array = np.array(lon_list, dtype=float)

    if np.sum(wind_u_array[:20] != -999) >= 7 and np.sum(wind_v_array[:20] != -999) >= 7:
        # Filter out NaN and invalid values
        valid_indices = (~np.isnan(pres_array) & ~np.isnan(temp_array) & ~np.isnan(dewp_array)
                         & (pres_array != -999) & (temp_array != -999) & (dewp_array != -999)
                         & (wind_v_array != -999) & (wind_u_array != -999))
        pres_valid = pres_array[valid_indices]
        temp_valid = temp_array[valid_indices]
        dewp_valid = dewp_array[valid_indices]
        wind_uu_knot = wind_u_array[valid_indices]
        wind_vv_knot = wind_v_array[valid_indices]
        # Convert wind speed from m/s to knots
        wind_uu_knots = wind_uu_knot * units('m/s').to('knots')
        wind_vv_knots = wind_vv_knot * units('m/s').to('knots')
        skew.plot_barbs(pres_valid[::100], wind_uu_knots[::100], wind_vv_knots[::100], y_clip_radius=0.03)
    else:
        # Filter out NaN and invalid values
        valid_indices = (~np.isnan(pres_array) & ~np.isnan(temp_array) & ~np.isnan(dewp_array)
                         & (pres_array != -999) & (temp_array != -999) & (dewp_array != -999))
        pres_valid = pres_array[valid_indices]
        temp_valid = temp_array[valid_indices]
        dewp_valid = dewp_array[valid_indices]

    valid_idx = (~np.isnan(lat_array) & ~np.isnan(lon_array) & (lat_array != -999) & (lon_array != -999))
    lat_valid = lat_array[valid_idx]
    lon_valid = lon_array[valid_idx]

    plt.xlabel(f"Temperature [{data['tdry_units']}]")
    plt.ylabel(f"Pressure [{data['pressure_units']}]")
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    skew.ax.scatter(temp_valid, pres_valid, color='tab:red', label='Temp.', s=20)
    skew.ax.scatter(dewp_valid, pres_valid, color='tab:blue', label='Dewpt.', s=20)
    temp_patch = Line2D([], [], color='tab:red', marker='o', linestyle='None', markersize=8, label='Temp.')
    dewpt_patch = Line2D([], [], color='tab:blue', marker='o', linestyle='None', markersize=8, label='Dewpt.')
    skew.ax.set_ylim(pres_valid.max(), min(pres_valid.min(), 400))
    skew.ax.set_xlim(min(-55, dewp_valid.min()), max(temp_valid.max(), dewp_valid.max(), 0))
    lcl_pressure, lcl_temperature = mpcalc.lcl(pres_valid[0] * units.hPa, temp_valid[0] * units.degC, dewp_valid[0] * units.degC)
    lfc_pres, lfc_temp = mpcalc.lfc(pres_valid * units.hPa, temp_valid * units.degC, dewp_valid * units.degC)

    parcel_prof = mpcalc.parcel_profile(pres_valid * units.hPa, temp_valid[0] * units.degC, dewp_valid[0] * units.degC)
    cape, cin = mpcalc.cape_cin(pres_valid * units.hPa, temp_valid * units.degC, dewp_valid * units.degC, parcel_prof)

    el_press, el_temp = mpcalc.el(pres_valid * units.hPa, temp_valid * units.degC, dewp_valid * units.degC, parcel_prof)
    el_point = skew.plot(el_press, el_temp, color='green', marker='_', ms=100, label='EL')

    skew.plot(pres_valid, parcel_prof, 'k', linewidth=2, label='Parcel Profile')
    lfc_point = skew.plot(lfc_pres, lfc_temp, color='magenta', marker='_', ms=100, label='LFC')
    lcl_point = skew.plot(lcl_pressure, lcl_temperature, color='cyan', marker='_', ms=100, label='LCL')

    skew.shade_cin(pres_valid * units.hPa, temp_valid * units.degC, parcel_prof, dewp_valid * units.degC)
    skew.shade_cape(pres_valid * units.hPa, temp_valid * units.degC, parcel_prof)

    cape_patch = mpatches.Patch(color='red', alpha=0.5, label=f'CAPE: {cape.magnitude:.2f} J/kg')
    cin_patch = mpatches.Patch(color='blue', alpha=0.5, label=f'CIN: {cin.magnitude:.2f} J/kg')
    parcel_patch = Line2D([], [], color='k', linewidth=2, label='Parcel Profile')
    el_patch = Line2D([], [], color='green', marker='_', linestyle='None', markersize=10, label='EL')
    lfc_patch = Line2D([], [], color='magenta', marker='_', linestyle='None', markersize=10, label='LFC')
    lcl_patch = Line2D([], [], color='cyan', marker='_', linestyle='None', markersize=10, label='LCL')

    fig.tight_layout(rect=[0, 0.2, 1, 0.96])
    plt.legend(handles=[cape_patch, cin_patch, parcel_patch, el_patch, lfc_patch, lcl_patch, temp_patch, dewpt_patch],
               loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=4)
    if lat_valid.size > 0 and lon_valid.size > 0:
        first_lat = lat_valid[0]
        first_lon = lon_valid[0]
        plt.title(f"CAESAR {data['Flight']} Drop #{file_number}: {data['launch_time']} Lat: {first_lat:.3f}, Lon: {first_lon:.3f}")
    else:
        plt.title(f"CAESAR {data['Flight']} Drop #{file_number}: {data['launch_time']}")
    plt.savefig(os.path.join(output_dir, f"{file_number}.png"))

def save_graph(file, file_number):
    data = file_parser(file)
    skewT_comparison(data, file_number)


def generate_graph_for_all_file(files):
    for i, file in enumerate(files, start=1):
        save_graph(file, i)
        print(f"Generated graphs for {file} and saved combined image")


if __name__ == '__main__':
    caesar_data_dir = "/data2/caesar/maslowski/caesar_data"

    data_files = sorted([os.path.join(caesar_data_dir, f) for f in os.listdir(caesar_data_dir) if f.endswith(".nc")], key=natural_sort_key)
    generate_graph_for_all_file(data_files)

