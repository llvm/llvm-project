#!/usr/bin/env python3

"""
Usage: <path/to/input-directory> <path/to/output-directory>

This script is used when building LLDB.framework or LLDBRPC.framework. For each framework, local includes are converted to their respective framework includes.

This script is used in 2 ways:
1. It is used on header files that are copied into LLDB.framework. For these files, local LLDB includes are converted into framework includes, e.g. #include "lldb/API/SBDefines.h" -> #include <LLDB/SBDefines.h>.

2. It is used on header files for LLDBRPC.framework. For these files, includes of RPC common files will be converted to framework includes, e.g. #include <lldb-rpc/common/RPCCommon.h> -> #include <LLDBRPC/RPCCommon.h>. It will also change local includes to framework includes, e.g. #include "SBAddress.h" -> #include <LLDBRPC/SBAddress.h>
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

# RPC header regexes
RPC_COMMON_REGEX = re.compile(r"#include <lldb-rpc/common/(?P<include_filename>.*)>")
RPC_INCLUDE_FILENAME_REGEX = re.compile(r'#include "(?P<include_filename>.*)"')


def modify_rpc_includes(input_file_path, output_file_path):
    with open(input_file_path, "r") as input_file:
        lines = input_file.readlines()
        file_buffer = "".join(lines)
        with open(output_file_path, "w") as output_file:
            # Local includes must be changed to RPC framework level includes.
            # e.g. #include "SBDefines.h" -> #include <LLDBRPC/SBDefines.h>
            # Also, RPC common code includes must change to RPC framework level includes.
            # e.g. #include "lldb-rpc/common/RPCPublic.h" -> #include <LLDBRPC/RPCPublic.h>
            rpc_common_matches = RPC_COMMON_REGEX.finditer(file_buffer)
            rpc_include_filename_matches = RPC_INCLUDE_FILENAME_REGEX.finditer(
                file_buffer
            )
            for match in rpc_common_matches:
                file_buffer = re.sub(
                    match.group(),
                    r"#include <LLDBRPC/" + match.group("include_filename") + ">",
                    file_buffer,
                )
            for match in rpc_include_filename_matches:
                file_buffer = re.sub(
                    match.group(),
                    r"#include <LLDBRPC/" + match.group("include_filename") + ">",
                    file_buffer,
                )
            output_file.write(file_buffer)


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
    parser.add_argument("-f", "--framework", choices=["lldb_main", "lldb_rpc"])
    parser.add_argument("-i", "--input_file")
    parser.add_argument("-o", "--output_file")
    parser.add_argument("-p", "--unifdef_path")
    parser.add_argument(
        "unifdef_guards",
        nargs="+",
        type=str,
        help="Guards to be removed with unifdef. These must be specified in the same way as they would be when passed directly into unifdef.",
    )
    args = parser.parse_args()
    input_file_path = str(args.input_file)
    output_file_path = str(args.output_file)
    framework_version = args.framework
    unifdef_path = str(args.unifdef_path)
    # Prepend dashes to the list of guards passed in from the command line.
    # unifdef takes the guards to remove as arguments in their own right (e.g. -USWIG)
    # but passing them in with dashes for this script causes argparse to think that they're
    # arguments in and of themself, so they need to passed in without dashes.
    unifdef_guards = ["-" + guard for guard in args.unifdef_guards]

    # Create the framework's header dir if it doesn't already exist
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    if framework_version == "lldb_main":
        modify_main_includes(input_file_path, output_file_path)
    if framework_version == "lldb_rpc":
        modify_rpc_includes(input_file_path, output_file_path)
    # After the incldues have been modified, run unifdef on the headers to remove any guards
    # specified at the command line.
    remove_guards(output_file_path, unifdef_path, unifdef_guards)


if __name__ == "__main__":
    main()
