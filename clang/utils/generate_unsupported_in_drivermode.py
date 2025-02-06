#!/usr/bin/env python3

"""generate_unsupported_in_drivermode.py

This script generates Lit regression test files that validate that options are only exposed to intended driver modes.

The options and driver modes are parsed from Options.td, whose path should be provided on the command line.
See clang/include/clang/Driver/Options.td

The path to the TableGen executable can optionally be provided. Otherwise, the script will search for it.

The primary maintenance task for this script would be updating the expected return message for a driver mode if
there are changes over time. See the instantiations of DriverData, specifically the check_str.

Logic:
1) For each option, (records of class "Option"), and for each driver, (records of class "OptionVisibility")
    a. if the option's "Visibility" field includes the driver flavour, skip processing this option for this driver
    b. if the option is part of an option group, (the record has the "Group" property),
       and the group's "Visibility" field includes the driver flavour, skip processing this option for this driver
    c. otherwise this option is not supported by this driver flavour, and this pairing is saved for testing
2) For each unsupported pairing, generate a Lit RUN line, and a CHECK line to parse for expected output. Ex: "error: unknown argument"
"""

import shutil
import os
import json
import subprocess
from bisect import bisect_left
from dataclasses import dataclass
import argparse
import dataclasses
from itertools import batched

# Strings defined in Options.td for the various driver flavours. See "OptionVisibility"
VISIBILITY_CC1AS = "CC1AsOption"
VISIBILITY_CC1 = "CC1Option"
VISIBILITY_CL = "CLOption"
VISIBILITY_DXC = "DXCOption"
VISIBILITY_DEFAULT = "DefaultVis"
VISIBILITY_FC1 = "FC1Option"
VISIBILITY_FLANG = "FlangOption"

# Lit test prefix strings
SLASH_SLASH = "// "
EXCLAMATION = "! "

# Invalid usage of the driver options below causes unique output, so skip testing
exceptions_sequence = [
    "cc1",
    "cc1as",
]


class UnsupportedDriverOption:
    """Defines an unsupported driver-option combination
    driver: The driver string as defined by OptionVisibility in Options.td
    option: The option object from Options.td
    option_name: Corresponding string for an option. See "Name" for a given option in Options.td
    prefix: String that precedes the option. Ex. "-"
    """

    def __init__(self, driver, option, option_name, prefix):
        self.driver = driver
        self.option = option
        self.option_name = option_name
        self.prefix = prefix

    # For sorting
    def __len__(self):
        return len(self.option_name)

    def __lt__(self, other):
        return len(self.option_name) > len(other.option_name)


@dataclass
class DriverData:
    """Dataclass for data specific to each driver
    lit_cmd_prefix: The beginning string of the Lit command
    lit_cmd_options: Strings containing additional options for this driver
    visibility_str: The corresponding visibility string from OptionVisibility in Options.td
    lit_cmd_end: String at the end of the Lit command
    check_str: The string or regex to be sent to FileCheck
    supported_sequence: List of UnsupportedDriverOption objects for supported options
                        that are Kind *JOINED*, as defined in Options.td
    test_option_sequence: A list of all the prefix-option pairs that will be tested for this driver
    """

    lit_cmd_prefix: str
    lit_cmd_options: str
    visibility_str: str
    lit_cmd_end: str = " - < /dev/null 2>&1 | FileCheck -check-prefix=CHECK-COUNT-"
    check_str: str = "{{(unknown argument|n?N?o such file or directory)}}"
    supported_sequence: list[UnsupportedDriverOption] = dataclasses.field(
        default_factory=list
    )
    test_option_sequence: list[str] = dataclasses.field(default_factory=list)


def find_groups(options_dictionary, option):
    """Find the groups for a given option
    Note that groups can themselves be part of groups, hence the recursion

    For example, considering option "C", it has the following 'Group' list as defined by Options.td:
      "Group": {
        "def": "Preprocessor_Group",
        "kind": "def",
        "printable": "Preprocessor_Group"
      },
    Preprocessor_Group is itself part of CompileOnly_Group, so option C would be part of both groups
      "Group": {
        "def": "CompileOnly_Group",
        "kind": "def",
        "printable": "CompileOnly_Group"
      },

    options_dictionary: The converted Python dictionary from the Options.td json string
    option: The option object from Options.td

    Return: A set including the group found for the option
    """
    group_list = options_dictionary[option]["Group"]

    if group_list is None:
        return None
    found_group = group_list["def"]
    group_set = {found_group}

    sub_group_set = find_groups(options_dictionary, found_group)
    if sub_group_set is None:
        return group_set
    else:
        group_set.update(sub_group_set)
        return group_set


def get_visibility(option):
    """Get a list of drivers that a given option is exposed to
    option: The option object from Options.td
    Return: Set that contains the visibilities of the given option
    """
    visibility_set = set(())
    # Check for the option's explicit visibility
    for visibility in options_dictionary[option]["Visibility"]:
        if visibility is not None:
            visibility_set.add(visibility["def"])

    # Check for the option's group's visibility
    group_set = find_groups(options_dictionary, option)
    if group_set is not None:
        for group_name in group_set:
            for visibility in options_dictionary[group_name]["Visibility"]:
                visibility_set.add(visibility["def"])

    return visibility_set


def get_lit_test_note(test_visibility):
    """Return the note to be included at the start of the Lit test file
    test_visibility: Any VISIBILITY_* variable. VISIBILITY_FLANG will return the .f90 formatted test note.
    All other will return the .c formatted test note
    """
    test_prefix = EXCLAMATION if test_visibility == VISIBILITY_FLANG else SLASH_SLASH

    return (
        f"{test_prefix}NOTE: This lit test was automatically generated to validate "
        "unintentionally exposed arguments to various driver flavours.\n"
        f"{test_prefix}NOTE: To make changes, see llvm-project/clang/utils/generate_unsupported_in_drivermode.py"
        + " from which it was generated.\n"
        f"{test_prefix}NOTE: Regenerate this Lit test with the following:\n"
        f"{test_prefix}NOTE: python generate_unsupported_in_drivermode.py "
        + "llvm-project/clang/include/clang/Driver/Options.td --llvm-bin llvm-project/build/bin --llvm-tblgen llvm-tblgen\n\n"
    )


def write_lit_test(test_path, test_visibility, unsupported_list):
    """Write the Lit tests to file
    test_path: File write path
    test_visibility: VISIBILITY_DEFAULT or VISIBILITY_FLANG, which indicates whether to write
    to the main Lit test file or flang Lit test file respectively
    unsupported_list: List of UnsupportedDriverOption objects
    """
    # If each option is tested with its own run line, the Lit tests become quite large. Instead, test options in batches
    try:
        with open(test_path, "w") as lit_file:
            lit_file.write(get_lit_test_note(test_visibility))
            batch_size = 100

            for visibility, driver_data in driver_data_dict.items():
                is_flang_pair = (
                    visibility == VISIBILITY_FLANG or visibility == VISIBILITY_FC1
                )

                if (test_visibility == VISIBILITY_FLANG and not is_flang_pair) or (
                    test_visibility == VISIBILITY_DEFAULT and is_flang_pair
                ):
                    continue

                comment_str = EXCLAMATION if is_flang_pair else SLASH_SLASH
                last_batch_size = 0

                unflattened_option_data = list(
                    batched(driver_data.test_option_sequence, batch_size)
                )

                for batch in unflattened_option_data:
                    # Example run line: // RUN: not --crash %clang -cc1 -A -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
                    run_cmd = (
                        f"{comment_str}RUN: not " + driver_data.lit_cmd_prefix
                    )  # "// RUN: not --crash %clang -cc1 "

                    for option_str in batch:
                        run_cmd += option_str + " "  # "-A"

                    run_cmd += (
                        driver_data.lit_cmd_options  # "-x c++"
                        + driver_data.lit_cmd_end  # " - < /dev/null 2>&1 | FileCheck  -check-prefix=CC1OptionCHECK-COUNT-"
                        + str(len(batch))  # 100
                        + " %s\n\n"  # " %s"
                    )

                    lit_file.write(run_cmd)

                    last_batch_size = len(batch)

                # CHECK statements. Instead of writing custom CHECK statements for each RUN line, create two statements
                # per driver. One statement for a full batch, and a second for a partial batch.
                check_cmd_start = (
                    comment_str + visibility + "CHECK-COUNT-"
                )  # //CC1OptionCHECK-COUNT-
                check_cmd_end = (
                    ": " + driver_data.check_str + "\n"
                )  # ": {{(unknown argument|n?N?o such file or directory)}}"
                check_cmd_full_batch = (
                    check_cmd_start + str(batch_size) + check_cmd_end
                )  # "//CC1OptionCHECK-COUNT-100: {{(unknown argument|n?N?o such file or directory)}}"
                check_cmd_partial_batch = (
                    check_cmd_start + str(last_batch_size) + check_cmd_end + "\n"
                )  # "//CC1OptionCHECK-COUNT-22: {{(unknown argument|n?N?o such file or directory)}}"

                lit_file.write(check_cmd_full_batch + check_cmd_partial_batch)

    except (FileNotFoundError, PermissionError, OSError):
        raise IOError(f"Error opening {test_path}. Exiting")
    else:
        lit_file.close()


def validate_file(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"Invalid file provided: {path}")
    return path


# List of driver flavours
driver_sequence = []
# List of unsupported driver-option pairs
unsupported_sequence = []
# List of driver-option pairs that will be skipped due to overlapping supported and unsupported option names.
# See later comments for detail
skipped_sequence = []

# Parse arguments
parser = argparse.ArgumentParser(
    description="This script generates Lit regression test files that validate that options are only exposed to "
    "intended driver modes. "
    "The options and driver modes are parsed from Options.td."
)

parser.add_argument(
    "<path>/Options.td",
    type=validate_file,
    help="Path to Options.td file. Typically found under clang/include/clang/Driver/Options.td",
)
parser.add_argument(
    "--llvm-bin",
    help="llvm build tree bin directory path. Must be specified with --llvm-tblgen. Default path: llvm-project/build/bin",
)
parser.add_argument(
    "--llvm-tblgen",
    help="LLVM TableGen executable. If not included with --llvm-bin, the script will search for the llvm-tblgen executable",
)

args = vars(parser.parse_args())

tablegen = ""
arg_llvm_bin = args["llvm_bin"]
arg_llvm_tblgen = args["llvm_tblgen"]
if arg_llvm_bin is None or arg_llvm_tblgen is None:
    tablegen = shutil.which("llvm-tblgen")
else:
    tablegen = arg_llvm_bin + "/" + arg_llvm_tblgen

# Run TableGen to convert Options.td to json
options_json_str = subprocess.run(
    [
        tablegen,
        "-I",
        os.path.join(os.path.dirname(__file__), "../../llvm/include"),
        args["<path>/Options.td"],
        "-dump-json",
    ],
    stdout=subprocess.PIPE,
)
options_dictionary = json.loads(options_json_str.stdout.decode("utf-8"))

# Establish the dataclass objects for each driver
driver_cc1as = DriverData(
    "%clang -cc1as ",
    "",
    VISIBILITY_CC1AS,
    f" - < /dev/null 2>&1 | FileCheck -check-prefix={VISIBILITY_CC1AS}CHECK-COUNT-",
)
driver_cc1 = DriverData(
    "%clang -cc1 ",
    " -x c++",
    VISIBILITY_CC1,
    f" - < /dev/null 2>&1 | FileCheck -check-prefix={VISIBILITY_CC1}CHECK-COUNT-",
)
driver_cl = DriverData(
    "%clang_cl ",
    " -### /c /WX -Werror",
    VISIBILITY_CL,
    f" 2>&1 | FileCheck -check-prefix={VISIBILITY_CL}CHECK-COUNT-",
    "{{(unknown argument ignored in|no such file or directory|argument unused during compilation)}}",
)
driver_dxc = DriverData(
    "%clang_dxc ",
    " -### /T lib_6_7",
    VISIBILITY_DXC,
    f" 2>&1 | FileCheck -check-prefix={VISIBILITY_DXC}CHECK-COUNT-",
    "{{(unknown argument|no such file or directory|argument unused during compilation)}}",
)
driver_default = DriverData(
    "%clang ",
    " -### -x c++ -c",
    VISIBILITY_DEFAULT,
    f" - < /dev/null 2>&1 | FileCheck -check-prefix={VISIBILITY_DEFAULT}CHECK-COUNT-",
    "{{(unknown argument|unsupported option|argument unused|no such file or directory)}}",
)
driver_fc1 = DriverData(
    "%clang --driver-mode=flang -fc1 ",
    "",
    VISIBILITY_FC1,
    f" - < /dev/null 2>&1 | FileCheck -check-prefix={VISIBILITY_FC1}CHECK-COUNT-",
    "{{(unknown argument|no such file or directory|does not exist)}}",
)
driver_flang = DriverData(
    "%clang --driver-mode=flang ",
    " -### -x c++ -c",
    VISIBILITY_FLANG,
    f" - < /dev/null 2>&1 | FileCheck -check-prefix={VISIBILITY_FLANG}CHECK-COUNT-",
    "{{unknown argument|unsupported option|argument unused during compilation|invalid argument|no such file or directory}}",
)

driver_data_dict = {
    VISIBILITY_CC1AS: driver_cc1as,
    VISIBILITY_CC1: driver_cc1,
    VISIBILITY_CL: driver_cl,
    VISIBILITY_DXC: driver_dxc,
    VISIBILITY_DEFAULT: driver_default,
    VISIBILITY_FC1: driver_fc1,
    VISIBILITY_FLANG: driver_flang,
}

# Gather list of driver flavours
for visibility in options_dictionary["!instanceof"]["OptionVisibility"]:
    driver_sequence.append(visibility)

# Iterate the options list and find which drivers shouldn't be visible to each option
for option in options_dictionary["!instanceof"]["Option"]:
    kind = options_dictionary[option]["Kind"]["def"]
    tmp_visibility_set = set(())
    option_name = options_dictionary[option]["Name"]

    # There are a few conditions that make an option unsuitable to test in this script
    # Options of kind KIND_INPUT & KIND_UNKNOWN don't apply to this test. For example,
    # Option "INPUT" with name "<input>".
    if (
        option in exceptions_sequence
        or options_dictionary[option]["Name"] is None
        or kind == "KIND_INPUT"
        or kind == "KIND_UNKNOWN"
    ):
        continue

    # Get the correct option prefix
    prefixes = options_dictionary[option]["Prefixes"]
    prefix = ""
    if prefixes is not None and len(prefixes) > 0:
        # Assuming the first prefix is the preferred prefix
        prefix = prefixes[0]

    tmp_visibility_set.update(get_visibility(option))

    # Check visibility of direct and indirect aliases
    # A given option may list only one "primary" alias, but that alias
    # may be listed by other options as well, hence indirect aliases
    alias_sequence = options_dictionary["!instanceof"]["Alias"]

    if options_dictionary[option]["Alias"] is not None:
        primary_alias = options_dictionary[option]["Alias"]["def"]

        tmp_visibility_set.update(get_visibility(primary_alias))

        for alias in alias_sequence:
            if options_dictionary[alias]["Alias"]["def"] == primary_alias:
                tmp_visibility_set.update(get_visibility(alias))

    for alias in alias_sequence:
        if options_dictionary[alias]["Alias"]["def"] == option:
            tmp_visibility_set.update(get_visibility(alias))

    # *JOINED* options that are supported need to be saved for checking
    # which options cannot be validated with this script
    is_option_kind_joined = "JOINED" in kind

    # Append to the unsupported list, and the various supported lists
    for driver in driver_sequence:
        if driver not in tmp_visibility_set:
            unsupported_sequence.append(
                UnsupportedDriverOption(driver, option, option_name, prefix)
            )
        elif is_option_kind_joined:
            driver_data_dict[driver].supported_sequence.append(
                UnsupportedDriverOption(driver, option, option_name, prefix)
            )

# Sort the supported lists for the next block
for visibility, driver_data in driver_data_dict.items():
    driver_data.supported_sequence.sort(key=len, reverse=True)

# For a given driver, this script cannot generate tests for unsupported options whose option "Name" have a prefix that
# corresponds to a supported/visible option of Kind *JOINED*. These driver-option pairs are removed here.
# The reason is that those options will be parsed as if they were the corresponding prefixed options with a value,
# and thus no error would be triggered.
# Example: Option "O_flag" is not visible to FlangOption, but option "O" is visible to FlangOption.
#          Attempting to test this:
#            clang --driver-mode=flang -O_flag -### -x c++ -c - < /dev/null 2>&1
#          Will be interpreted as this:
#            clang --driver-mode=flang -O _flag -### -x c++ -c - < /dev/null 2>&1
for unsupported_pair in unsupported_sequence:
    supported_seq = driver_data_dict[unsupported_pair.driver].supported_sequence

    start_index = bisect_left(supported_seq, unsupported_pair)

    for supported_pair in supported_seq[start_index:]:
        if (
            unsupported_pair.option_name.startswith(supported_pair.option_name)
            and unsupported_pair not in skipped_sequence
        ):
            skipped_sequence.append(unsupported_pair)

for skip_pair in skipped_sequence:
    unsupported_sequence.remove(skip_pair)

# Add the final list of option data to each driver's test list
for index, unsupported_pair in enumerate(unsupported_sequence):
    driver_data_dict[unsupported_pair.driver].test_option_sequence.append(
        unsupported_pair.prefix + unsupported_pair.option_name
    )

write_lit_test(
    "../test/Driver/unsupported_in_drivermode.c",
    VISIBILITY_DEFAULT,
    unsupported_sequence,
)
write_lit_test(
    "../test/Driver/flang/unsupported_in_flang.f90",
    VISIBILITY_FLANG,
    unsupported_sequence,
)
