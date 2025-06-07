#!/usr/bin/env python3

"""generate_unsupported_in_drivermode.py

This script generates Lit regression test files that validate that options are only exposed to intended driver modes.

The options and driver modes are parsed from Options.td, whose path should be provided on the command line.
See clang/include/clang/Driver/Options.td

The LLVM TableGen executable can optionally be provided along with the path to the LLVM build tree bin directory.
Otherwise, the script will search for it.

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

import os
import json
import subprocess
from bisect import bisect_left
from dataclasses import dataclass
import argparse
import dataclasses
import itertools

# Strings defined in Options.td for the various driver flavours. See "OptionVisibility"
VISIBILITY_CC1AS = "CC1AsOption"
VISIBILITY_CC1 = "CC1Option"
VISIBILITY_CL = "CLOption"
VISIBILITY_DXC = "DXCOption"
VISIBILITY_DEFAULT = "DefaultVis"
VISIBILITY_FC1 = "FC1Option"
VISIBILITY_FLANG = "FlangOption"

exceptions_sequence = [
    # Invalid usage of the driver options below causes unique output, so skip testing
    "cc1",
    "cc1as",
    # There is currently a bug with "_no_warnings", i.e. --no-warnings. Diagnostic related options
    # are parsed first, and always with CC1 visibility. They're used to set up the diagnostic
    # engine, which parses "_no_warnings" (and its alias "w", i.e. -w) and sets an internal flag
    # that suppresses all warnings.
    "_no_warnings",
    "w",
]


@dataclass
class DriverOption:
    """Defines an unsupported driver-option combination
    driver: The driver string as defined by OptionVisibility in Options.td
    option: The option object from Options.td
    option_name: Corresponding string for an option. See "Name" for a given option in Options.td
    prefix: String that precedes the option. Ex. "-"
    """

    driver: str
    option: str
    option_name: str
    prefix: str


@dataclass
class DriverData:
    """Dataclass for data specific to each driver
    lit_cmd_prefix: The beginning string of the Lit command
    lit_cmd_options: Strings containing additional options for this driver
    visibility_str: The corresponding visibility string from OptionVisibility in Options.td
    lit_cmd_end: String at the end of the Lit command
    check_str: The string or regex to be sent to FileCheck
    supported_joined_option_sequence: List of DriverOption objects for supported options
                                      that are Kind *JOINED*, as defined in Options.td
    supported_non_joined_option_sequence: List of DriverOption objects for supported options
                                          that are not Kind *JOINED*, as defined in Options.td
    test_option_sequence: A list of all the prefix-option pairs that will be tested for this driver
    """

    lit_cmd_prefix: str
    lit_cmd_options: str
    visibility_str: str
    lit_cmd_end: str
    check_str: str
    check_prefix: str
    supported_joined_option_sequence: list[DriverOption] = dataclasses.field(
        default_factory=list
    )
    supported_non_joined_option_sequence: list[DriverOption] = dataclasses.field(
        default_factory=list
    )
    test_option_sequence: list[str] = dataclasses.field(default_factory=list)


def collect_transitive_groups(member, options_dictionary):
    """Find the groups for a given member, where a member can be an option or a group.
    Note that groups can themselves be part of groups, hence the recursion

    For example, considering option 'C', it has the following 'Group' field as defined by Options.td:
      "C": {
        "Group": {
          "def": "Preprocessor_Group",
          // ...
        },
        // ...
      },
    'Preprocessor_Group' is itself part of 'CompileOnly_Group', so option 'C' would be part of both groups
      "Preprocessor_Group": {
        // ...
        "Group": {
          "def": "CompileOnly_Group",
          // ...
        },
        // ...
      },

    member: An option object or group object from Options.td.
    options_dictionary: The converted Python dictionary from the Options.td json string

    Return: A set including the group(s) found for the member. If no groups found, returns an empty set
    """
    parent_field = options_dictionary[member]["Group"]
    if parent_field is None:
        return set()

    parent_name = parent_field["def"]
    return {parent_name} | collect_transitive_groups(parent_name, options_dictionary)


def get_visibility(option):
    """Get a list of drivers that a given option is exposed to
    option: The option object from Options.td
    Return: Set that contains the visibilities of the given option
    """
    visibility_set = set()
    # Check for the option's explicit visibility
    for visibility in options_dictionary[option]["Visibility"]:
        if visibility is not None:
            visibility_set.add(visibility["def"])

    # Check for the option's group's visibility
    for group_name in collect_transitive_groups(option, options_dictionary):
        for visibility in options_dictionary[group_name]["Visibility"]:
            visibility_set.add(visibility["def"])

    return visibility_set


def get_lit_test_note(test_visibility):
    """Return the note to be included at the start of the Lit test file
    test_visibility: Any VISIBILITY_* variable. VISIBILITY_DEFAULT will return the .c formatted test note.
    All other will return the .f90 formatted test note
    """
    test_prefix = "// " if test_visibility == VISIBILITY_DEFAULT else "! "

    return (
        f"{test_prefix}NOTE: This lit test was automatically generated to validate "
        "unintentionally exposed arguments to various driver flavours.\n"
        f"{test_prefix}NOTE: To make changes, see llvm-project/clang/utils/generate_unsupported_in_drivermode.py"
        + " from which it was generated.\n"
        f"{test_prefix}NOTE: Regenerate this Lit test with the following:\n"
        f"{test_prefix}NOTE: python generate_unsupported_in_drivermode.py "
        "--options-td-dir llvm-project/clang/include/clang/Driver --llvm-include-dir llvm-project/llvm/include --llvm-tblgen llvm-project/build/bin/llvm-tblgen\n\n"
    )


def write_lit_test(test_path, test_visibility):
    """Write the Lit tests to file
    test_path: File write path
    test_visibility: VISIBILITY_DEFAULT, VISIBILITY_FLANG, or VISIBILITY_FC1 which indicates whether to write
    to the main Lit test file, the flang test file, or the flang -fc1 test file
    """
    lit_file = open(test_path, "w")

    lit_file.write(get_lit_test_note(test_visibility))
    batch_size = 100

    for visibility, driver_data in driver_data_dict.items():
        is_flang_pair = visibility == VISIBILITY_FLANG or visibility == VISIBILITY_FC1

        if (
            (test_visibility == VISIBILITY_FLANG and visibility != VISIBILITY_FLANG)
            or (test_visibility == VISIBILITY_FC1 and visibility != VISIBILITY_FC1)
            or (test_visibility == VISIBILITY_DEFAULT and is_flang_pair)
        ):
            continue

        comment_str = "! " if is_flang_pair else "// "

        batched = itertools.batched(driver_data.test_option_sequence, batch_size)
        for i, batch in enumerate(batched):
            # Example run line: "// RUN: not %clang -cc1 -A ... -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK0 %s"
            run_cmd = (
                f"\n{comment_str}RUN: not " + driver_data.lit_cmd_prefix
            )  # "// RUN: not %clang -cc1 "

            # // RUN: <command up to this point> \
            # // RUN:   --one-option \
            # // RUN:   -a-different-option \
            # ...
            run_cmd += f" \\\n{comment_str}RUN:   ".join(itertools.chain(("",), batch))

            run_cmd += (
                driver_data.lit_cmd_options  # "-x c++"
                + driver_data.lit_cmd_end  # " - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK"
                + str(i)  # "0"
                + " %s\n\n"  # " %s"
            )

            lit_file.write(run_cmd)

            for option_str in batch:
                # Example check line: "// CHECK_CC1_0: {{(unknown argument).*}}-A"
                check_cmd = (
                    comment_str  # "//
                    + driver_data.check_prefix  # "CHECK_CC1_"
                    + str(i)  # "0"
                    + ": {{("
                    + driver_data.check_str  # "unknown argument"
                    + ").*"
                    + "}}"
                    + option_str  # "-A"
                    + "\n"
                )
                lit_file.write(check_cmd)

    lit_file.close()


# List of driver flavours
driver_sequence = []
# List of unsupported driver-option pairs
unsupported_sequence = []
# List of driver-option pairs that will be skipped due to overlapping supported and unsupported option names.
# See later comments for detail
skipped_sequence = []

# Parse arguments
parser = argparse.ArgumentParser(
    description="Generate a lit test to validate that options are only exposed to "
    "intended driver modes. "
    "The options and driver modes are parsed from Options.td."
)

default_options_file = os.path.join(
    os.path.dirname(__file__), "../include/clang/Driver/Options.td"
)
parser.add_argument(
    "--options-file",
    help="Path to Options.td. Typically found under clang/include/clang/Driver/Options.td, this is the default.",
    default=default_options_file,
)
default_llvm_include_dir = os.path.join(os.path.dirname(__file__), "../../llvm/include")
parser.add_argument(
    "--llvm-include-dir",
    help="Include directory for LLVM TableGen executable. The typical source tree path will be used by default.",
    default=default_llvm_include_dir,
)
parser.add_argument(
    "--llvm-bin",
    help="llvm build tree bin directory path. Only used if --llvm-tblgen is not specified.",
)
parser.add_argument(
    "--llvm-tblgen",
    help="LLVM TableGen executable. If --llvm-bin is also not specified, the PATH will be searched.",
)
parser.add_argument(
    "--general-test-path",
    help="File where most driver flavour tests will be written. The typical source tree path will be used by default.",
)
parser.add_argument(
    "--flang-test-path",
    help="File where flang tests will be written. The typical source tree path will be used by default.",
)
parser.add_argument(
    "--flang-fc1-test-path",
    help="File where flang -fc1 tests will be written. The typical source tree path will be used by default.",
)
args = parser.parse_args()

if not args.llvm_tblgen:
    if args.llvm_bin:
        args.llvm_tblgen = os.path.join(args.llvm_bin, "llvm-tblgen")
    else:
        args.llvm_tblgen = "llvm-tblgen"

# Run TableGen to convert Options.td to json
# Ex: llvm-tblgen -I llvm-project/llvm/include llvm-project/clang/include/clang/Driver/Options.td -dump-json
options_json_str = subprocess.run(
    [
        args.llvm_tblgen,
        "-I",
        args.llvm_include_dir,
        args.options_file,
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
    f" - < /dev/null 2>&1 | FileCheck -check-prefix=CHECK_CC1AS_",
    "unknown argument",
    "CHECK_CC1AS_",
)
driver_cc1 = DriverData(
    "%clang -cc1 ",
    " -x c++",
    VISIBILITY_CC1,
    f" - < /dev/null 2>&1 | FileCheck -check-prefix=CHECK_CC1_",
    "unknown argument",
    "CHECK_CC1_",
)
driver_cl = DriverData(
    "%clang_cl ",
    " -### /c /WX -Werror",
    VISIBILITY_CL,
    f" 2>&1 | FileCheck -check-prefix=CHECK_CL_",
    "unknown argument ignored in clang-cl",
    "CHECK_CL_",
)
driver_dxc = DriverData(
    "%clang_dxc ",
    " -### /T lib_6_7",
    VISIBILITY_DXC,
    f" 2>&1 | FileCheck -check-prefix=CHECK_DXC_",
    "unknown argument",
    "CHECK_DXC_",
)
driver_default = DriverData(
    "%clang ",
    " -### -x c++ -c",
    VISIBILITY_DEFAULT,
    f" - < /dev/null 2>&1 | FileCheck -check-prefix=CHECK_CLANG_",
    "unknown argument",
    "CHECK_CLANG_",
)
driver_fc1 = DriverData(
    "%flang_fc1 ",
    "",
    VISIBILITY_FC1,
    f" - < /dev/null 2>&1 | FileCheck -check-prefix=CHECK_FC1_",
    "unknown argument",
    "CHECK_FC1_",
)
driver_flang = DriverData(
    "%clang --driver-mode=flang ",
    " -### -x c++ -c",
    VISIBILITY_FLANG,
    f" - < /dev/null 2>&1 | FileCheck -check-prefix=CHECK_FLANG_",
    "unknown argument",
    "CHECK_FLANG_",
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
    option_name = options_dictionary[option]["Name"]

    # There are a few conditions that make an option unsuitable to test in this script
    # Options of kind KIND_INPUT & KIND_UNKNOWN don't apply to this test. For example,
    # Option "INPUT" with name "<input>".
    if (
        options_dictionary[option]["Name"] is None
        or kind == "KIND_INPUT"
        or kind == "KIND_UNKNOWN"
    ):
        continue

    # Get the correct option prefix
    prefixes = options_dictionary[option]["Prefixes"]
    prefix = "-"
    if prefixes is not None and len(prefixes) > 0:
        # Assuming the first prefix is the preferred prefix
        prefix = prefixes[0]

    visibility_set = get_visibility(option)

    # *JOINED* options that are supported need to be saved for checking
    # which options cannot be validated with this script
    is_option_kind_joined = "JOINED" in kind

    # Append to the unsupported list, and the various supported lists
    for driver in driver_sequence:
        if driver not in visibility_set and option not in exceptions_sequence:
            unsupported_sequence.append(
                DriverOption(driver, option, option_name, prefix)
            )
        elif is_option_kind_joined:
            driver_data_dict[driver].supported_joined_option_sequence.append(
                DriverOption(driver, option, option_name, prefix)
            )
        else:
            driver_data_dict[driver].supported_non_joined_option_sequence.append(
                DriverOption(driver, option, option_name, prefix)
            )

# Sort the supported lists for the next block
for driver_data in driver_data_dict.values():
    driver_data.supported_joined_option_sequence.sort(
        key=lambda o: len(o.option_name), reverse=True
    )
    driver_data.supported_non_joined_option_sequence.sort(
        key=lambda o: len(o.option_name), reverse=True
    )

# For a given driver, this script cannot generate tests for unsupported options whose option "Name" have a prefix that
# corresponds to a supported/visible option of Kind *JOINED*. These driver-option pairs are removed here.
# The reason is that those options will be parsed as if they were the corresponding prefixed options with a value,
# and thus no error would be triggered.
# Example: Option "O_flag" is not visible to FlangOption, but option "O" is visible to FlangOption.
#          Attempting to test this:
#            clang --driver-mode=flang -O_flag -### -x c++ -c - < /dev/null 2>&1
#          Will be interpreted as this:
#            clang --driver-mode=flang -O _flag -### -x c++ -c - < /dev/null 2>&1
#
# Additionally, there are certain distinct options with matching option names, which would otherwise be distinguished by
# the prefix used. Unfortunately, as documented earlier, this script replaces prefix "/" with "-". As a result, the
# visibility of corresponding options must be considered.
# Example: Option "_SLASH_H" is not visible to CC1Option, but option "H" is visible to CC1Option.
for unsupported_pair in unsupported_sequence:
    supported_joined_seq = driver_data_dict[
        unsupported_pair.driver
    ].supported_joined_option_sequence
    supported_non_joined_seq = driver_data_dict[
        unsupported_pair.driver
    ].supported_non_joined_option_sequence

    # Check for matching kind *JOINED* option name prefixes
    start_index = bisect_left(
        supported_joined_seq,
        -len(unsupported_pair.option_name),
        key=lambda o: -len(o.option_name),
    )

    for supported_pair in supported_joined_seq[start_index:]:
        if (
            unsupported_pair.option_name.startswith(supported_pair.option_name)
            and unsupported_pair not in skipped_sequence
        ):
            skipped_sequence.append(unsupported_pair)

    # Check for matching option names
    start_index = bisect_left(
        supported_non_joined_seq,
        -len(unsupported_pair.option_name),
        key=lambda o: -len(o.option_name),
    )

    for supported_pair in supported_non_joined_seq[start_index:]:
        if (
            unsupported_pair.option_name == supported_pair.option_name
            and unsupported_pair.driver == supported_pair.driver
            and unsupported_pair not in skipped_sequence
        ):
            skipped_sequence.append(unsupported_pair)
            break
        if len(supported_pair.option_name) > len(unsupported_pair.option_name):
            break

for skip_pair in skipped_sequence:
    unsupported_sequence.remove(skip_pair)

# Add the final list of option data to each driver's test list
for unsupported_pair in unsupported_sequence:
    # When options prefixed with "/" are used with unsupported drivers, misleading output is returned that also
    # makes parsing more complicated. Instead, given all "/" prefix options accept prefix "-" as well, use "-",
    # which returns the typical error.
    # Example:
    #   clang -cc1 /AI -x c++
    #     error: error reading '/AI': No such file or directory
    #   clang -cc1 -AI -x c++
    #     error: unknown argument: '-AI'
    unsupported_pair.prefix = (
        "-" if unsupported_pair.prefix == "/" else unsupported_pair.prefix
    )

    driver_data_dict[unsupported_pair.driver].test_option_sequence.append(
        unsupported_pair.prefix + unsupported_pair.option_name
    )

args.general_test_path = (
    "../test/Driver/unsupported_in_drivermode.c"
    if not args.general_test_path
    else args.general_test_path
)
args.flang_test_path = (
    "../test/Driver/flang/unsupported_in_flang.f90"
    if not args.flang_test_path
    else args.flang_test_path
)
args.flang_fc1_test_path = (
    "../../flang/test/Driver/unsupported_in_flang_fc1.f90"
    if not args.flang_fc1_test_path
    else args.flang_fc1_test_path
)

write_lit_test(
    args.general_test_path,
    VISIBILITY_DEFAULT,
)
write_lit_test(
    args.flang_test_path,
    VISIBILITY_FLANG,
)
write_lit_test(
    args.flang_fc1_test_path,
    VISIBILITY_FC1,
)
