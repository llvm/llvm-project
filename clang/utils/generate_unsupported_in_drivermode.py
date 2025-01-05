#!/usr/bin/env python3

""" generate_unsupported_in_drivermode.py

usage: python generate_unsupported_in_drivermode.py <path>/Options.td [<path>/llvm-tblgen]

This script generates a Lit regression test file that validates that options
are only exposed to intended driver modes.

The options and driver modes are parsed from Options.td, whose path should be
provided on the command line. See clang/include/clang/Driver/Options.td

The path to the TableGen executable can optionally be provided. Otherwise, the
script will search for it.

Logic:
1) For each option, (records of class "Option"), and for each driver, (records of class "OptionVisibility")
    a. if the option's "Visibility" field includes the driver flavour, skip processing this option for this driver
    b. if the option is part of an option group, (the record has the "Group" property),
       and the group's "Visibility" field includes the driver flavor, skip processing this option for this driver
    c. otherwise this option is not supported by this driver flavor, and this pairing is saved for testing
2) For each unsupported pairing, generate a Lit RUN line, and a CHECK line to parse for expected output. Ex: "error: unknown argument"
"""

import sys
import shutil
import os
import json
import subprocess
import math
from pathlib import Path

LLVM_TABLEGEN = "llvm-tblgen"
LIT_TEST_PATH = "../test/Driver/Inputs/unsupported-driver-options-check.ll"
INCLUDE_PATH = "../../llvm/include"

# Strings defined in Options.td for the various driver flavours. See "OptionVisibility"
VISIBILITY_CC1AS = "CC1AsOption"
VISIBILITY_CC1 = "CC1Option"
VISIBILITY_CL = "CLOption"
VISIBILITY_DXC = "DXCOption"
VISIBILITY_DEFAULT = "DefaultVis"
VISIBILITY_FC1 = "FC1Option"
VISIBILITY_FLANG = "FlangOption"

# Strings used in the commands to be tested
CLANG = "clang"
CLANG_CL = "clang-cl"
CLANG_DXC = "clang-dxc"
FLANG = "flang-new"
OPTION_NUM = "-###"
OPTION_X = "-x"
OPTION_CPP = "c++"
OPTION_C = "-c"
LIT_CMD_END = " - < /dev/null 2>&1 | FileCheck %s\n"

# See clang/include/clang/Basic/DiagnosticDriverKinds.td for the *unknown_argument* strings
# As per Driver::ParseArgStrings from Driver.cpp, all the driver modes use the
# string "unknown argument" in their unsupported option error messages
ERROR_MSG_CHECK = ("{{(unknown argument|"
                     "argument unused|"
                     "unsupported|"
                     "unknown integrated tool)}}")

LIT_TEST_NOTE = ("; NOTE: This lit test was automatically generated to validate "
                 "unintentionally exposed arguments to various driver flavours.\n"
                 "; NOTE: To make changes, see " + Path(__file__).resolve().as_posix()
                 + " from which it was generated.\n"
                 "To output which unsupported options are not tested by this Lit"
                 " test, see that script\n\n")

exceptions_sequence = ["Wno_rewrite_macros", # Default
                       "fexperimental_sanitize_metadata_EQ_atomics", # Default
                       "fexperimental_sanitize_metadata_EQ_covered", # Default
                       "fexperimental_sanitize_metadata_EQ_uar", # Default
                       "mno_strict_align", # CC1
                       "mstrict_align",
                       "fheinous-gnu-extensions",
                       "fcuda-approx-transcendentals"] # CC1 TODO: This is temporary

class DriverController:
    """ Controller for data specific to each driver
    shell_cmd_prefix: The beginning string of the command to be tested
    visibility_str: The corresponding visibility string from OptionVisibility in Options.td
    shell_cmd_suffix: Strings near the end of the command to be tested
    supported_sequence: List of UnsupportedDriverOption objects for supported options
                        that are Kind KIND_JOINED*, as defined in Options.td
    is_os_compatible: Boolean indicating whether this driver is available on the current OS
    """
    def __init__(self, shell_cmd_prefix = "", visibility_str = "", shell_cmd_suffix = "", is_os_compatible = False):
        self.shell_cmd_prefix = shell_cmd_prefix
        self.visibility_str = visibility_str
        self.shell_cmd_suffix = shell_cmd_suffix
        self.supported_sequence = []
        self.is_os_compatible = is_os_compatible

class UnsupportedDriverOption:
    """ Defines an unsupported driver-option combination
    driver: The driver string as defined by OptionVisibility in Options.td
    option: The option string. See "Name" for a given option in Options.td
    prefix: String that precedes the option. Ex. "-"
    is_error: Boolean indicating whether the corresponding command generates an error
    """
    def __init__(self, driver, option, prefix):
        self.driver = driver
        self.option = option
        self.prefix = prefix
        self.is_error = True

    # For sorting
    def __len__(self):
        return len(self.option)

def print_usage():
    """ Print valid usage of this script
    """
    sys.exit( "usage: python " + sys.argv[0] + " <path>/Options.td [<path>/llvm-tblgen]" )

def find_file(file_name, search_path):
    """ Find the given file name under a search path
    """
    result = []

    for root, dir, files in os.walk(search_path):
        if file_name in files:
            result.append(os.path.join(root, file_name))
    return result

def is_valid_file(path, expected_name):
    """ Is a file valid
    Check if a given path is to a file, and if it matches the expected file name
    """
    if path.is_file() and path.name == expected_name:
        return True
    else:
        return False

def find_executable(executable):
    """ Validate an executable
    """
    result = shutil.which(executable)
    if result is None:
        print(f"Unable to find {executable}")
    else:
        print(f"{executable} found: {result}")

    return result

def find_tablegen():
    """ Validate the TableGen executable
    """
    result = find_executable(LLVM_TABLEGEN)
    if result is None:
        sys.exit("\nExiting")
    else:
        return result

def find_groups(group_sequence, options_json, option):
    """ Find the groups for a given option
    Note that groups can themselves be part of groups, hence the recursion
    """
    group_json = options_json[option]["Group"]

    if group_json is None:
        return

    # Prevent circular group membership lookup
    for group in group_sequence:
        if group_json["def"] == group:
            return

    group_sequence.append(group_json["def"])
    return find_groups(group_sequence, options_json, option)

# Validate the number of arguments have been passed
argc = len(sys.argv)
if argc < 2 or argc > 3:
    print_usage()

options_input_path = Path(sys.argv[1])
tablegen_input_path = ""
tablegen = None
options_td = ""
driver_sequence = []
options_sequence = []
unsupported_sequence = []
# List of driver-option pairs that will be skipped due to
# overlapping supported and unsupported option names. See later comments for detail
skipped_sequence = []
# List of driver-option pairs that will be skipped due to
# a variety of limitations. See usage for detail
untested_sequence = []

current_path = os.path.dirname(__file__)

# Validate Options.td
if not is_valid_file(options_input_path, "Options.td"):
    print("Invalid Options.td path. Searching for valid path...")

    relative_path = "../"
    search_path = os.path.join(current_path, relative_path)

    file_search_list = find_file("Options.td", search_path)
    if len(file_search_list) != 1:
        print_usage()
        sys.exit("Unable to find Options.td.\nExiting")
    else:
        options_td = file_search_list[0]
        print(options_td)
else:
    options_td = options_input_path.resolve().as_posix()

# Validate TableGen executable
if argc > 2:
    tablegen_input_path = Path(sys.argv[2])
    if not is_valid_file(tablegen_input_path, "llvm-tblgen"):
        print("Invalid tablegen path. Searching for valid path...")
        tablegen = find_tablegen()
    else:
        tablegen = tablegen_input_path.resolve().as_posix()
else:
    tablegen = find_tablegen()

# Run TableGen to convert Options.td to json
options_json_str = subprocess.run([ tablegen, "-I", os.path.join(current_path, INCLUDE_PATH), options_td, "-dump-json"], stdout=subprocess.PIPE)
options_json = json.loads(options_json_str.stdout.decode('utf-8'))

# Establish the controller objects for each driver
driver_cc1as = DriverController(f"{CLANG} -cc1as", VISIBILITY_CC1AS, "", None != find_executable(CLANG))
driver_cc1 = DriverController(f"{CLANG} -cc1", VISIBILITY_CC1, " " + OPTION_X + " " + OPTION_CPP, None != find_executable(CLANG))
driver_cl = DriverController(CLANG_CL, VISIBILITY_CL, " " + OPTION_NUM + " " + OPTION_X + " " + OPTION_CPP + " " + OPTION_C, None != find_executable(CLANG_CL))
driver_dxc = DriverController(CLANG_DXC, VISIBILITY_DXC, " " + OPTION_NUM + " " + OPTION_X + " " + OPTION_CPP + " " + OPTION_C, None != find_executable(CLANG_DXC))
driver_default = DriverController(CLANG, VISIBILITY_DEFAULT, " " + OPTION_NUM + " " + OPTION_X + " " + OPTION_CPP + " " + OPTION_C, None != find_executable(CLANG))
driver_fc1 = DriverController(f"{FLANG} -fc1", VISIBILITY_FC1, "", None != find_executable(FLANG))
driver_flang = DriverController(FLANG, VISIBILITY_FLANG, " " + OPTION_NUM + " " + OPTION_X + " " + OPTION_CPP + " " + OPTION_C, None != find_executable(FLANG))

driver_controller = [driver_cc1as, driver_cc1, driver_cl, driver_dxc, driver_default, driver_fc1, driver_flang]

def get_index(driver_vis):
    """ Get the driver controller index for a given driver
    driver_vis: The visibility string from OptionVisibility in Options.td
    """
    for index, driver_ctrl in enumerate(driver_controller):
        if driver_vis == driver_ctrl.visibility_str:
            return index

# Gather list of driver flavours
for visibility in options_json["!instanceof"]["OptionVisibility"]:
    driver_sequence.append(visibility)

# Walk through the options list and find which drivers shouldn't be visible to each option
for option in options_json["!instanceof"]["Option"]:
    kind = options_json[option]["Kind"]["def"]
    should_skip = False
    tmp_vis_list = []
    group_sequence = []
    option_name = options_json[option]["Name"]

    # There are a few conditions that make an option unsuitable to test in this script
    # Options of kind KIND_INPUT & KIND_UNKNOWN don't apply to this test. For example,
    # Option "INPUT" with name "<input>".
    if option in exceptions_sequence or \
        options_json[option]["Name"] is None or \
        kind == "KIND_INPUT" or \
        kind == "KIND_UNKNOWN":

        untested_sequence.append(UnsupportedDriverOption("All", option, ""))
        continue

    # Get the correct option prefix
    prefixes = options_json[option]["Prefixes"]
    prefix = ""
    if prefixes is not None and len(prefixes) > 0:
        # Assuming the first prefix is the preferred prefix
        prefix = prefixes[0]
        if os.name != "nt" and prefix == "/":
            continue

    # Check for the option's explicit visibility
    for visibility in options_json[option]["Visibility"]:
        if visibility is not None:
            tmp_vis_list.append(visibility["def"])

    # Check for the option's group's visibility
    find_groups(group_sequence, options_json, option)
    if len(group_sequence) > 0:
        for group_name in group_sequence:
            # For clang_ignored_f_Group & f_Group see description in Options.td
            # "Temporary groups for clang options which we know we don't support,
            # but don't want to verbosely warn the user about."
            if group_name == "clang_ignored_f_Group" or group_name == "f_Group":
                should_skip = True
                break
            for visibility in options_json[group_name]["Visibility"]:
                tmp_vis_list.append(visibility["def"])
    if should_skip:
        untested_sequence.append(UnsupportedDriverOption("All", option, ""))
        continue

    # KIND_JOINED* options that are supported need to be saved for checking
    # which options cannot be validated with this script
    is_option_kind_joined = kind == "KIND_JOINED" or kind == "KIND_JOINED_OR_SEPARATE"

    # Append to the unsupported list, and the various supported lists
    for driver in driver_sequence:
        if driver not in tmp_vis_list:
            unsupported_sequence.append(UnsupportedDriverOption(driver, option_name, prefix))
        elif is_option_kind_joined:
            driver_controller[get_index(driver)].supported_sequence.append(UnsupportedDriverOption(driver, option_name, prefix))

def find_supported_seq_cmp_start(supported_sequence, low, high, search_option):
    """ Return the index where to start comparisons in the supported sequence
    Modified binary search for the first element of supported_sequence
    that has an option that is of equal or lesser length than the search option
    from the unsupported sequence
    The supported sequence must be reverse sorted by option name length
    """
    middle = math.floor(low + (high - low) / 2)

    if low > high:
        return -1
    # If the start of the list is reached
    if middle - 1 == -1:
        return middle
    # If the end of the list is reached
    if middle == len(supported_sequence)-1:
        return middle

    if len(supported_sequence[middle].option) <= len(search_option) < len(supported_sequence[middle - 1].option):
        return middle
    elif len(supported_sequence[middle].option) <= len(search_option):
        return find_supported_seq_cmp_start(supported_sequence, low, middle - 1, search_option)
    elif len(supported_sequence[middle].option) > len(search_option):
        return find_supported_seq_cmp_start(supported_sequence, middle+1, high, search_option)
    else:
        # No-op
        return -1

# Sort the supported lists for the next block
for driver_ctrl in driver_controller:
    driver_ctrl.supported_sequence.sort(key=len, reverse=True)

# For a given driver, this script cannot generate tests for unsupported options
# that have a prefix that is a supported option of Kind KIND_JOINED*.
# These driver-option pairs are removed here.
for unsupported_pair in unsupported_sequence:
    supported_seq = driver_controller[get_index(unsupported_pair.driver)].supported_sequence
    start_index = find_supported_seq_cmp_start(supported_seq, 0, len(supported_seq)-1, unsupported_pair.option)
    start_index = 0 if start_index == -1 else start_index

    for supported_pair in driver_controller[get_index(unsupported_pair.driver)].supported_sequence[start_index:]:
        if unsupported_pair.option.startswith(supported_pair.option):
            skipped_sequence.append(unsupported_pair)

for skip_pair in skipped_sequence:
    unsupported_sequence.remove(skip_pair)

# Preprocess each default driver command to determine if they result in an error status or a warning
# This is necessary since the Lit tests require an explicit "; RUN: not" for errors
for unsupported_pair in unsupported_sequence:
    if (driver_controller[get_index(unsupported_pair.driver)].is_os_compatible and
            driver_controller[get_index(unsupported_pair.driver)].visibility_str == VISIBILITY_DEFAULT):
        # Run each command inside the script
        cmd = [f"{driver_controller[get_index(unsupported_pair.driver)].shell_cmd_prefix} \
                 {unsupported_pair.prefix}{unsupported_pair.option} \
                 {driver_controller[get_index(unsupported_pair.driver)].shell_cmd_suffix} -"]
        cmd_out = subprocess.run( cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        unsupported_pair.is_error = True if cmd_out.returncode == 1 else False

# Generate the Lit test
try:
    with open(LIT_TEST_PATH, "w") as lit_file:
        try:
            lit_file.write(LIT_TEST_NOTE)

            for unsupported_pair in unsupported_sequence:
                if unsupported_pair.is_error:
                    lit_not = "not "
                else:
                    lit_not = ""

                CMD_START = "; RUN: " + lit_not

                if driver_controller[get_index(unsupported_pair.driver)].is_os_compatible:
                    lit_file.write(
                        CMD_START +
                        driver_controller[get_index(unsupported_pair.driver)].shell_cmd_prefix +
                        " " +
                        unsupported_pair.prefix +
                        unsupported_pair.option +
                        driver_controller[get_index(unsupported_pair.driver)].shell_cmd_suffix +
                        LIT_CMD_END)
            lit_file.write("; CHECK: " + ERROR_MSG_CHECK + "\n")
        except(IOError, OSError):
            sys.exit("Error writing to " + "LIT_TEST_PATH. Exiting")
except(FileNotFoundError, PermissionError, OSError):
    sys.exit("Error opening " + "LIT_TEST_PATH" + ". Exiting")
else:
    lit_file.close()

# print("\nThese unsupported driver-option pairs were not tested:")
# for untested_pair in untested_sequence:
#     print(f"Driver: {untested_pair.driver}\tOption:{untested_pair.option}")
# for skipped_pair in skipped_sequence:
#     print(f"Driver: {skipped_pair.driver}\tOption:{skipped_pair.option}")
