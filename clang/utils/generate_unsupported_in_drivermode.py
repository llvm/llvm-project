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
from pathlib import Path

LLVM_TABLEGEN = "llvm-tblgen"
LIT_TEST_PATH = "../test/Driver/Inputs/unsupported-driver-options-check.ll"
INCLUDE_PATH = "../../llvm/include"
PREFIX = "CHECK-"

# Strings used in Options.td for various driver flavours
OPTION_CC1AS = "CC1AsOption"
OPTION_CC1 = "CC1Option"
OPTION_CL = "CLOption"
OPTION_DXC = "DXCOption"
OPTION_DEFAULT = "DefaultVis"
OPTION_FC1 = "FC1Option"
OPTION_FLANG = "FlangOption"

# Error messages output from each driver
ERROR_MSG_CC1AS = ": error: unknown argument"
ERROR_MSG_CC1 = "error: unknown argument"
ERROR_MSG_CL = "" # TODO
ERROR_MSG_DXC = "" # TODO
ERROR_MSG_DEFAULT = "clang: error: unknown argument"
ERROR_MSG_FC1 = "error: unknown argument"
ERROR_MSG_FLANG = "flang: error: unknown argument"

# Lit CHECK prefixes
CHECK_PREFIX_CC1AS = PREFIX + OPTION_CC1AS
CHECK_PREFIX_CC1 = PREFIX + OPTION_CC1
CHECK_PREFIX_CL = PREFIX + OPTION_CL
CHECK_PREFIX_DXC = PREFIX + OPTION_DXC
CHECK_PREFIX_DEFAULT = PREFIX + OPTION_DEFAULT
CHECK_PREFIX_FC1 = PREFIX + OPTION_FC1
CHECK_PREFIX_FLANG = PREFIX + OPTION_FLANG

LIT_TEST_NOTE = ("; NOTE: This lit test was automatically generated to validate " +
                 "unintentionally exposed arguments to various driver flavours.\n"
                 "; NOTE: To make changes, see " + Path(__file__).resolve().as_posix()
                 + " from which it was generated.\n\n")

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

def find_tablegen():
    """ Validate the TableGen executable
    """
    result = shutil.which(LLVM_TABLEGEN)
    if result is None:
        sys.exit("Unable to find " + LLVM_TABLEGEN + ".\nExiting")
    else:
        print("TableGen found: " + result)
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


class UnsupportedDriverOption():
    def __init__(self, driver, option):
        self.driver = driver
        self.option = option

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

# Gather list of driver flavours
for i in options_json["!instanceof"]["OptionVisibility"]:
    driver_sequence.append(i)

# Gather list of options
for i in options_json["!instanceof"]["Option"]:
    options_sequence.append(i)

# Walk through the options list and find which drivers shouldn't be visible to each option
for option in options_sequence:
    tmp_vis_list = []
    group_sequence = []

    # Check for the option's explicit visibility
    for visibility in options_json[option]["Visibility"]:
        tmp_vis_list.append(visibility["def"])

    # Check for the option's group's visibility
    find_groups(group_sequence, options_json, option)
    if len(group_sequence) > 0:
        for group_name in group_sequence:
            for visibility in options_json[group_name]["Visibility"]:
                tmp_vis_list.append(visibility["def"])

    # Append to the unsupported list
    for driver in driver_sequence:
        if driver not in tmp_vis_list:
            unsupported_sequence.append(UnsupportedDriverOption(driver, option))

# Generate the lit test
try:
    with open(LIT_TEST_PATH, "w") as lit_file:
        try:
            lit_file.write(LIT_TEST_NOTE)

            for i in unsupported_sequence:
                if i.driver == OPTION_CC1AS:
                    lit_file.write(
                        "; RUN: not clang -cc1as -" + i.option + " -help 2>&1 | FileCheck -check-prefix=" + CHECK_PREFIX_CC1AS + " %s\n")
                    continue
                if i.driver == OPTION_CC1:
                    lit_file.write(
                        "; RUN: not clang -cc1 -" + i.option + " -help 2>&1 | FileCheck -check-prefix=" + CHECK_PREFIX_CC1 + " %s\n")
                    continue
                # if i.driver == OPTION_CL:
                #     lit_file.write(
                #         "; RUN: not clang-cl -" + i.option + " -help 2>&1 | FileCheck -check-prefix=" + CHECK_PREFIX_CL + " %s\n")
                #     continue
                # if i.driver == OPTION_DXC:
                #     lit_file.write(
                #         "; RUN: not clang-dxc -" + i.option + " -help 2>&1 | FileCheck -check-prefix=" + CHECK_PREFIX_DXC + " %s\n")
                #     continue
                if i.driver == OPTION_DEFAULT:
                    lit_file.write(
                        "; RUN: not clang -" + i.option + " -help 2>&1 | FileCheck -check-prefix=" + CHECK_PREFIX_DEFAULT + " %s\n")
                    continue
                if i.driver == OPTION_FC1:
                    lit_file.write(
                        "; RUN: not flang -fc1 -" + i.option + " -help 2>&1 | FileCheck -check-prefix=" + CHECK_PREFIX_FC1 + " %s\n")
                    continue
                if i.driver == OPTION_FLANG:
                    lit_file.write(
                        "; RUN: not flang -" + i.option + " -help 2>&1 | FileCheck -check-prefix=" + CHECK_PREFIX_FLANG + " %s\n")

            lit_file.write("; " + CHECK_PREFIX_CC1AS + ": " + ERROR_MSG_CC1AS + "\n")
            lit_file.write("; " + CHECK_PREFIX_CC1 + ": " + ERROR_MSG_CC1 + "\n")
            lit_file.write("; " + CHECK_PREFIX_CL + ": " + ERROR_MSG_CL + "\n")
            lit_file.write("; " + CHECK_PREFIX_DXC + ": " + ERROR_MSG_DXC + "\n")
            lit_file.write("; " + CHECK_PREFIX_DEFAULT + ": " + ERROR_MSG_DEFAULT + "\n")
            lit_file.write("; " + CHECK_PREFIX_FC1 + ": " + ERROR_MSG_FC1 + "\n")
            lit_file.write("; " + CHECK_PREFIX_FLANG + ": " + ERROR_MSG_FLANG + "\n")
        except(IOError, OSError):
            sys.exit("Error writing to " + "LIT_TEST_PATH. Exiting")
except(FileNotFoundError, PermissionError, OSError):
    sys.exit("Error opening " + "LIT_TEST_PATH" + ". Exiting")
else:
    lit_file.close()