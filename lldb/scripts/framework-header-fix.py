#!/usr/bin/env python3

"""
Usage: <path/to/input-directory> <path/to/output-directory>

This script is used when building LLDB.framework. For each framework, local includes are converted to their respective framework includes.

This script is used used on header files that are copied into LLDB.framework. For these files, local LLDB includes are converted into framework includes, e.g. #include "lldb/API/SBDefines.h" -> #include <LLDB/SBDefines.h>.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

# Main header regexes
INCLUDE_FILENAME_REGEX = re.compile(
    r'#include "lldb/(API/)?(?P<include_filename>.*){0,1}"'
)

def modify_main_includes(input_file_path, output_file_path):
    with open(input_file_path, "r") as input_file:
        lines = input_file.readlines()
        file_buffer = "".join(lines)
        with open(output_file_path, "w") as output_file:
            # Local includes must be changed to framework level includes.
            # e.g. #include "lldb/API/SBDefines.h" -> #include <LLDB/SBDefines.h>
            regex_matches = INCLUDE_FILENAME_REGEX.finditer(file_buffer)
            for match in regex_matches:
                file_buffer = re.sub(
                    match.group(),
                    r"#include <LLDB/" + match.group("include_filename") + ">",
                    file_buffer,
                )
            output_file.write(file_buffer)


def remove_guards(output_file_path, unifdef_path, unifdef_guards):
    # The unifdef path should be passed in from CMake. If it wasn't there in CMake or is incorrect,
    # find it using shutil. If shutil can't find it, then exit.
    if not shutil.which(unifdef_path):
        unifdef_path = shutil.which("unifdef")
    if not unifdef_path:
        print(
            "Unable to find unifdef executable. Guards will not be removed from input files. Exiting..."
        )
        sys.exit()

    subprocess_command = (
        [unifdef_path, "-o", output_file_path] + unifdef_guards + [output_file_path]
    )
    subprocess.run(subprocess_command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file")
    parser.add_argument("-o", "--output_file")
    parser.add_argument("-p", "--unifdef_path")
    parser.add_argument(
        "--unifdef_guards",
        nargs="+",
        type=str,
        help="Guards to be removed with unifdef. These must be specified in the same way as they would be when passed directly into unifdef.",
    )
    args = parser.parse_args()
    input_file_path = str(args.input_file)
    output_file_path = str(args.output_file)
    unifdef_path = str(args.unifdef_path)
    # Prepend dashes to the list of guards passed in from the command line.
    # unifdef takes the guards to remove as arguments in their own right (e.g. -USWIG)
    # but passing them in with dashes for this script causes argparse to think that they're
    # arguments in and of themself, so they need to passed in without dashes.
    if args.unifdef_guards:
        unifdef_guards = ["-U" + guard for guard in args.unifdef_guards]

    # Create the framework's header dir if it doesn't already exist
    try:
        os.makedirs(os.path.dirname(output_file_path))
    except FileExistsError:
        pass

    modify_main_includes(input_file_path, output_file_path)
    # After the incldues have been modified, run unifdef on the headers to remove any guards
    # specified at the command line.
    if args.unifdef_guards:
        remove_guards(output_file_path, unifdef_path, unifdef_guards)


if __name__ == "__main__":
    main()
