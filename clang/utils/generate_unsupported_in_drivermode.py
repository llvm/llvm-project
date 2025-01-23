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
LIT_TEST_PATH = "../test/Driver/unsupported_in_drivermode.c"
LIT_TEST_PATH_FLANG = "../test/Driver/flang/unsupported_in_flang.f90"
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
CLANG_CL = f"{CLANG} --driver-mode=cl"
CLANG_DXC = f"{CLANG} --driver-mode=dxc"
FLANG = f"{CLANG} --driver-mode=flang"
CLANG_LIT = "%clang"
CLANG_CL_LIT = "%clang_cl"
CLANG_DXC_LIT = "%clang_dxc"
FLANG_LIT = f"%{FLANG}"
OPTION_HASH = "-###"
OPTION_X = "-x"
OPTION_WX = "/WX"
OPTION_CPP = "c++"
OPTION_C = "-c"
OPTION_CC1 = "-cc1"
OPTION_CC1AS = "-cc1as"
OPTION_FC1 = "-fc1"
OPTION_SLASH_C = "/c"
OPTION_T = "/T lib_6_7"
SLASH_SLASH = "// "
EXCLAMATION = "! "

# A few options need to be explicitly skipped for a variety of reasons
exceptions_sequence = [
    # Invalid usage of the driver options below causes unique output
    "cc1",
    "cc1as",
]


class DriverController:
    """Controller for data specific to each driver
    shell_cmd_prefix: The beginning string of the command to be tested
    lit_cmd_prefix: The beginning string of the Lit command
    visibility_str: The corresponding visibility string from OptionVisibility in Options.td
    shell_cmd_suffix: Strings near the end of the command to be tested
    check_string: The string or regex to be sent to FileCheck
    lit_cmd_end: String at the end of the Lit command

    supported_sequence: List of UnsupportedDriverOption objects for supported options
                        that are Kind KIND_JOINED*, as defined in Options.td
    """

    def __init__(
        self,
        shell_cmd_prefix="",
        lit_cmd_prefix="",
        visibility_str="",
        shell_cmd_suffix="",
        check_string="{{(unknown argument|n?N?o such file or directory)}}",
        lit_cmd_end=" - < /dev/null 2>&1 | FileCheck -check-prefix=",
    ):
        self.shell_cmd_prefix = shell_cmd_prefix
        self.lit_cmd_prefix = lit_cmd_prefix
        self.visibility_str = visibility_str
        self.shell_cmd_suffix = shell_cmd_suffix
        self.supported_sequence = []
        self.check_string = check_string
        self.lit_cmd_end = lit_cmd_end


class UnsupportedDriverOption:
    """Defines an unsupported driver-option combination
    driver: The driver string as defined by OptionVisibility in Options.td
    option: The option object from Options.td
    option_name: Corresponding string for an option. See "Name" for a given option in Options.td
    prefix: String that precedes the option. Ex. "-"
    is_error: Boolean indicating whether the corresponding command generates an error
    """

    def __init__(self, driver, option, option_name, prefix):
        self.driver = driver
        self.option = option
        self.option_name = option_name
        self.prefix = prefix
        self.is_error = True

    # For sorting
    def __len__(self):
        return len(self.option_name)


def print_usage():
    """Print valid usage of this script"""
    sys.exit("usage: python " + sys.argv[0] + " <path>/Options.td [<path>/llvm-tblgen]")


def find_file(file_name, search_path):
    """Find the given file name under a search path"""
    result = []

    for root, dir, files in os.walk(search_path):
        if file_name in files:
            result.append(os.path.join(root, file_name))
    return result


def is_valid_file(path, expected_name):
    """Is a file valid
    Check if a given path is to a file, and if it matches the expected file name
    """
    if path.is_file() and path.name == expected_name:
        return True
    else:
        return False


def find_tablegen():
    """Validate the TableGen executable"""
    result = shutil.which(LLVM_TABLEGEN)
    if result is None:
        print(f"Unable to find {LLVM_TABLEGEN}")
        sys.exit("\nExiting")
    else:
        print(f"{LLVM_TABLEGEN} found: {result}")
        return result


def find_groups(group_sequence, options_json, option):
    """Find the groups for a given option
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
options_json_str = subprocess.run(
    [
        tablegen,
        "-I",
        os.path.join(current_path, INCLUDE_PATH),
        options_td,
        "-dump-json",
    ],
    stdout=subprocess.PIPE,
)
options_json = json.loads(options_json_str.stdout.decode("utf-8"))

# Establish the controller objects for each driver
driver_cc1as = DriverController(
    f"{CLANG} {OPTION_CC1AS}",
    f"{CLANG_LIT} {OPTION_CC1AS}",
    VISIBILITY_CC1AS,
    "",
)
driver_cc1 = DriverController(
    f"{CLANG} {OPTION_CC1}",
    f"{CLANG_LIT} {OPTION_CC1}",
    VISIBILITY_CC1,
    " " + OPTION_X + " " + OPTION_CPP,
)
driver_cl = DriverController(
    CLANG_CL,
    CLANG_CL_LIT,
    VISIBILITY_CL,
    " " + OPTION_HASH + " " + OPTION_SLASH_C + " " + OPTION_WX,
    "{{(unknown argument ignored in|no such file or directory|argument unused during compilation)}}",
    " 2>&1 | FileCheck -check-prefix=",
)
driver_dxc = DriverController(
    CLANG_DXC,
    CLANG_DXC_LIT,
    VISIBILITY_DXC,
    " " + OPTION_HASH + " " + OPTION_T,
    "{{(unknown argument|no such file or directory|argument unused during compilation)}}",
    " 2>&1 | FileCheck -check-prefix=",
)
driver_default = DriverController(
    CLANG,
    CLANG_LIT,
    VISIBILITY_DEFAULT,
    " " + OPTION_HASH + " " + OPTION_X + " " + OPTION_CPP + " " + OPTION_C,
    "{{(unknown argument|unsupported option|argument unused|no such file or directory)}}",
)
driver_fc1 = DriverController(
    f"{FLANG} {OPTION_FC1}",
    f"{FLANG_LIT} {OPTION_FC1}",
    VISIBILITY_FC1,
    "",
    "{{(unknown argument|no such file or directory|does not exist)}}",
)
# As per flang.f90, "-fc1 is invoked when in --driver-mode=flang",
# so no point including the below.
# driver_flang = DriverController(
#     FLANG,
#     FLANG_LIT,
#     VISIBILITY_FLANG,
#     " " + OPTION_HASH + " " + OPTION_X + " " + OPTION_CPP + " " + OPTION_C,
#     "{{unknown argument|unsupported option|argument unused during compilation|invalid argument|no such file or directory}}",
# )

driver_controller = [
    driver_cc1as,
    driver_cc1,
    driver_cl,
    driver_dxc,
    driver_default,
    driver_fc1,
    # driver_flang,
]


def get_index(driver_vis):
    """Get the driver controller index for a given driver
    driver_vis: The visibility string from OptionVisibility in Options.td
    """
    for index, driver_ctrl in enumerate(driver_controller):
        if driver_vis == driver_ctrl.visibility_str:
            return index


def get_visibility(option, filtered_visibility):
    """Get a list of drivers that a given option exposed to
    option: The option object from Options.td
    filtered_visibility: Sequence in which the visibility will be stored

    Return true if this option should be skipped
    """
    group_sequence = []
    should_skip = False

    # Check for the option's explicit visibility
    for visibility in options_json[option]["Visibility"]:
        if visibility is not None:
            filtered_visibility.append(visibility["def"])

    # Check for the option's group's visibility
    find_groups(group_sequence, options_json, option)
    if len(group_sequence) > 0:
        for group_name in group_sequence:
            for visibility in options_json[group_name]["Visibility"]:
                filtered_visibility.append(visibility["def"])
    if should_skip:
        untested_sequence.append(
            UnsupportedDriverOption("All", option, options_json[option]["Name"], "")
        )

    return should_skip


def find_supported_seq_cmp_start(supported_sequence, low, high, search_option):
    """Return the index where to start comparisons in the supported sequence
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
    if middle == len(supported_sequence) - 1:
        return middle

    if (
        len(supported_sequence[middle].option_name)
        <= len(search_option)
        < len(supported_sequence[middle - 1].option_name)
    ):
        return middle
    elif len(supported_sequence[middle].option_name) <= len(search_option):
        return find_supported_seq_cmp_start(
            supported_sequence, low, middle - 1, search_option
        )
    elif len(supported_sequence[middle].option_name) > len(search_option):
        return find_supported_seq_cmp_start(
            supported_sequence, middle + 1, high, search_option
        )
    else:
        # No-op
        return -1


def get_lit_test_note(test_visibility):
    """Return the note to be included at the start of the Lit test file"""
    test_prefix = EXCLAMATION if test_visibility == VISIBILITY_FLANG else SLASH_SLASH

    return (
        f"{test_prefix}UNSUPPORTED: system-windows\n"
        f"{test_prefix}NOTE: This lit test was automatically generated to validate "
        "unintentionally exposed arguments to various driver flavours.\n"
        f"{test_prefix}NOTE: To make changes, see "
        + Path(__file__).resolve().as_posix()
        + " from which it was generated.\n\n"
    )


def write_lit_test(test_path, test_visibility, unsupported_list):
    """Write the lit tests to file"""
    try:
        with open(test_path, "w") as lit_file:
            try:
                lit_file.write(get_lit_test_note(test_visibility))

                for index, unsupported_pair in enumerate(unsupported_list):
                    is_flang_pair = (
                        unsupported_pair.driver == VISIBILITY_FLANG
                        or unsupported_pair.driver == VISIBILITY_FC1
                    )
                    if (test_visibility == VISIBILITY_FLANG and not is_flang_pair) or (
                        test_visibility == VISIBILITY_DEFAULT and is_flang_pair
                    ):
                        continue

                    # In testing, return codes cannot be relied on for consistently for assessing command failure.
                    # Leaving this handling here in case things change, but for now, Lit tests will accept pass or fail
                    # lit_not = "not " if unsupported_pair.is_error else ""

                    lit_not = "not not --crash "

                    prefix_str = SLASH_SLASH
                    if (
                        unsupported_pair.driver == VISIBILITY_FLANG
                        or unsupported_pair.driver == VISIBILITY_FC1
                    ):
                        prefix_str = EXCLAMATION

                    CMD_START = f"{prefix_str}RUN: " + lit_not

                    lit_file.write(
                        CMD_START
                        + driver_controller[
                            get_index(unsupported_pair.driver)
                        ].lit_cmd_prefix
                        + " "
                        + unsupported_pair.prefix
                        + unsupported_pair.option_name
                        + driver_controller[
                            get_index(unsupported_pair.driver)
                        ].shell_cmd_suffix
                        + driver_controller[
                            get_index(unsupported_pair.driver)
                        ].lit_cmd_end
                        + unsupported_pair.driver
                        + " %s\n"
                    )
                # CHECK statements. Instead of writing custom CHECK statements for each option-driver pair,
                # create one statement per driver. Not all options return error messages that include their option name
                for driver in driver_controller:
                    is_flang_driver = (
                        driver.visibility_str == VISIBILITY_FLANG
                        or driver.visibility_str == VISIBILITY_FC1
                    )

                    if test_visibility == VISIBILITY_FLANG and not is_flang_driver:
                        continue
                    elif test_visibility == VISIBILITY_DEFAULT and is_flang_driver:
                        continue

                    check_prefix = EXCLAMATION if is_flang_driver else SLASH_SLASH

                    lit_file.write(
                        check_prefix
                        + driver.visibility_str
                        + ": "
                        + driver.check_string
                        + "\n"
                    )
            except (IOError, OSError):
                sys.exit("Error writing to " + "LIT_TEST_PATH. Exiting")
    except (FileNotFoundError, PermissionError, OSError):
        sys.exit("Error opening " + "LIT_TEST_PATH" + ". Exiting")
    else:
        lit_file.close()


# Gather list of driver flavours
for visibility in options_json["!instanceof"]["OptionVisibility"]:
    if visibility == VISIBILITY_FLANG:
        continue
    driver_sequence.append(visibility)

# Iterate the options list and find which drivers shouldn't be visible to each option
for option in options_json["!instanceof"]["Option"]:
    kind = options_json[option]["Kind"]["def"]
    should_skip = False
    tmp_vis_list = []
    group_sequence = []
    option_name = options_json[option]["Name"]

    # There are a few conditions that make an option unsuitable to test in this script
    # Options of kind KIND_INPUT & KIND_UNKNOWN don't apply to this test. For example,
    # Option "INPUT" with name "<input>".
    if (
        option_name in exceptions_sequence
        or options_json[option]["Name"] is None
        or kind == "KIND_INPUT"
        or kind == "KIND_UNKNOWN"
    ):
        untested_sequence.append(
            UnsupportedDriverOption("All", option, option_name, "")
        )
        continue

    # Get the correct option prefix
    prefixes = options_json[option]["Prefixes"]
    prefix = ""
    if prefixes is not None and len(prefixes) > 0:
        # Assuming the first prefix is the preferred prefix
        prefix = prefixes[0]

    should_skip = get_visibility(option, tmp_vis_list)

    # Check visibility of direct and indirect aliases
    # A given option may list only one "primary" alias, but that alias
    # may be listed by other options as well, hence indirect aliases
    alias_sequence = options_json["!instanceof"]["Alias"]

    if options_json[option]["Alias"] is not None:
        primary_alias = options_json[option]["Alias"]["def"]

        should_skip = get_visibility(primary_alias, tmp_vis_list)

        for alias in alias_sequence:
            if options_json[alias]["Alias"]["def"] == primary_alias:
                should_skip = get_visibility(alias, tmp_vis_list)

    for alias in alias_sequence:
        if options_json[alias]["Alias"]["def"] == option:
            should_skip = get_visibility(alias, tmp_vis_list)

    if should_skip:
        continue

    # KIND_JOINED* options that are supported need to be saved for checking
    # which options cannot be validated with this script
    is_option_kind_joined = kind == "KIND_JOINED" or kind == "KIND_JOINED_OR_SEPARATE"

    # Append to the unsupported list, and the various supported lists
    for driver in driver_sequence:
        if driver not in tmp_vis_list:
            unsupported_sequence.append(
                UnsupportedDriverOption(driver, option, option_name, prefix)
            )
        elif is_option_kind_joined:
            driver_controller[get_index(driver)].supported_sequence.append(
                UnsupportedDriverOption(driver, option, option_name, prefix)
            )

# Sort the supported lists for the next block
for driver_ctrl in driver_controller:
    driver_ctrl.supported_sequence.sort(key=len, reverse=True)

# For a given driver, this script should not generate tests for unsupported options
# whose option Name have a prefix that corresponds to a supported option / visible option of Kind KIND_JOINED*.
# These driver-option pairs are removed here.
for unsupported_pair in unsupported_sequence:
    supported_seq = driver_controller[
        get_index(unsupported_pair.driver)
    ].supported_sequence

    start_index = find_supported_seq_cmp_start(
        supported_seq, 0, len(supported_seq) - 1, unsupported_pair.option_name
    )
    start_index = 0 if start_index == -1 else start_index

    for supported_pair in driver_controller[
        get_index(unsupported_pair.driver)
    ].supported_sequence[start_index:]:
        if (
            unsupported_pair.option_name.startswith(supported_pair.option_name)
            and unsupported_pair not in skipped_sequence
        ):
            skipped_sequence.append(unsupported_pair)

for skip_pair in skipped_sequence:
    unsupported_sequence.remove(skip_pair)
skipped_sequence.clear()

# Preprocess each default driver command to determine if they result in an error status or a warning.
# The other drivers currently output error for all unsupported commands, so preprocessing is unnecessary
# This is necessary since the Lit tests require an explicit "; RUN: not" for errors
for unsupported_pair in unsupported_sequence:
    if (
        driver_controller[get_index(unsupported_pair.driver)].visibility_str
        == VISIBILITY_DEFAULT
    ):
        # Run each command inside the script
        cmd = [
            f"{driver_controller[get_index(unsupported_pair.driver)].shell_cmd_prefix} \
                 {unsupported_pair.prefix}{unsupported_pair.option_name} \
                 {driver_controller[get_index(unsupported_pair.driver)].shell_cmd_suffix} -"
        ]

        tmp_file = "tmp_file.txt"
        # Open a temporary file in binary mode since some stderr output may trigger decoding errors
        with open(tmp_file, "wb+") as out_file:
            cmd_out = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=out_file,
                stderr=subprocess.STDOUT,
                shell=True,
            )

            unsupported_pair.is_error = True if cmd_out.returncode == 1 else False

            # Options corresponding to driver flavours may be added automatically, in which case,
            # their visibility should be considered as well.
            tmp_vis_list = []
            get_visibility(unsupported_pair.option, tmp_vis_list)
            out_file.seek(0)
            out = out_file.read()
            if b"-cc1" in out and VISIBILITY_CC1 in tmp_vis_list:
                skipped_sequence.append(unsupported_pair)
            elif b"-cc1as" in out and VISIBILITY_CC1AS in tmp_vis_list:
                skipped_sequence.append(unsupported_pair)

        os.remove(tmp_file)

for skip_pair in skipped_sequence:
    unsupported_sequence.remove(skip_pair)
skipped_sequence.clear()

write_lit_test(LIT_TEST_PATH, VISIBILITY_DEFAULT, unsupported_sequence)
write_lit_test(LIT_TEST_PATH_FLANG, VISIBILITY_FLANG, unsupported_sequence)
