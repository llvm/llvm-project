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
import subprocess

# Main header regexes
INCLUDE_FILENAME_REGEX = re.compile(
    r'#include "lldb/API/(?P<include_filename>.*){0,1}"'
)

# RPC header regexes
RPC_COMMON_REGEX = re.compile(r"#include <lldb-rpc/common/(?P<include_filename>.*)>")
RPC_INCLUDE_FILENAME_REGEX = re.compile(r'#include "(?P<include_filename>.*)"')


def modify_rpc_includes(input_directory_path, output_directory_path):
    for input_filepath in os.listdir(input_directory_path):
        current_input_file = os.path.join(input_directory_path, input_filepath)
        output_dest = os.path.join(output_directory_path, input_filepath)
        if os.path.isfile(current_input_file):
            with open(current_input_file, "r") as input_file:
                lines = input_file.readlines()
                file_buffer = "".join(lines)
            with open(output_dest, "w") as output_file:
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


def modify_main_includes(input_directory_path, output_directory_path):
    for input_filepath in os.listdir(input_directory_path):
        current_input_file = os.path.join(input_directory_path, input_filepath)
        output_dest = os.path.join(output_directory_path, input_filepath)
        if os.path.isfile(current_input_file):
            with open(current_input_file, "r") as input_file:
                lines = input_file.readlines()
                file_buffer = "".join(lines)
            with open(output_dest, "w") as output_file:
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


def remove_guards(output_directory_path, unifdef_path, unifdef_guards):
    # The unifdef path should be passed in from CMake. If it wasn't there in CMake,
    # find it using shutil.
    if not unifdef_path:
        unifdef_path = shutil.which("unifdef")
    for current_file in os.listdir(output_directory_path):
        if (os.path.isfile(current_file)):
            current_file = os.path.join(output_directory_path, current_file)
            subprocess_command = (
                [unifdef_path, "-o", current_file] + unifdef_guards + [current_file]
            )
            subprocess.run(subprocess_command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--framework", choices=["lldb_main", "lldb_rpc"])
    parser.add_argument("-i", "--input_directory")
    parser.add_argument("-o", "--output_directory")
    parser.add_argument("-p", "--unifdef_path")
    parser.add_argument(
        "unifdef_guards",
        nargs="+",
        type=str,
        help="Guards to be removed with unifdef. These must be specified in the same way as they would be when passed directly into unifdef.",
    )
    args = parser.parse_args()
    input_directory_path = str(args.input_directory)
    output_directory_path = str(args.output_directory)
    framework_version = args.framework
    unifdef_path = str(args.unifdef_path)
    # Prepend dashes to the list of guards passed in from the command line.
    # unifdef takes the guards to remove as arguments in their own right (e.g. -USWIG)
    # but passing them in with dashes for this script causes argparse to think that they're
    # arguments in and of themself, so they need to passed in without dashes.
    unifdef_guards = ["-" + guard for guard in args.unifdef_guards]

    if framework_version == "lldb_main":
        modify_main_includes(input_directory_path, output_directory_path)
    if framework_version == "lldb_rpc":
        modify_rpc_includes(input_directory_path, output_directory_path)
    # After the incldues have been modified, run unifdef on the headers to remove any guards
    # specified at the command line.
    remove_guards(output_directory_path, unifdef_path, unifdef_guards)


if __name__ == "__main__":
    main()
