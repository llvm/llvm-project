#!/usr/bin/env python3
# Usage: framework-header-version-fix.py <path/to/input-header.h> <path/to/output-header.h> MAJOR MINOR PATCH
# This script modifies lldb-rpc-defines.h to uncomment the macro defines used for the LLDB
# major, minor and patch values as well as populating their definitions.

import argparse
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("lldb_version_major")
    parser.add_argument("lldb_version_minor")
    parser.add_argument("lldb_version_patch")
    args = parser.parse_args()
    input_path = str(args.input)
    output_path = str(args.output)
    lldb_version_major = args.lldb_version_major
    lldb_version_minor = args.lldb_version_minor
    lldb_version_patch = args.lldb_version_patch

    with open(input_path, "r") as input_file:
        lines = input_file.readlines()

    with open(output_path, "w") as output_file:
        for line in lines:
            # Uncomment the line that defines the LLDB major version and populate its value.
            if re.match(r"//#define LLDB_RPC_VERSION$", line):
                output_file.write(
                    re.sub(
                        r"//#define LLDB_RPC_VERSION",
                        r"#define LLDB_RPC_VERSION " + lldb_version_major,
                        line,
                    )
                )
            # Uncomment the line that defines the LLDB minor version and populate its value.
            elif re.match(r"//#define LLDB_RPC_REVISION$", line):
                output_file.write(
                    re.sub(
                        r"//#define LLDB_RPC_REVISION",
                        r"#define LLDB_RPC_REVISION " + lldb_version_minor,
                        line,
                    )
                )
            # Uncomment the line that defines the complete LLDB version string and populate its value.
            elif re.match(r"//#define LLDB_RPC_VERSION_STRING$", line):
                output_file.write(
                    re.sub(
                        r"//#define LLDB_RPC_VERSION_STRING",
                        r'#define LLDB_RPC_VERSION_STRING "{0}.{1}.{2}"'.format(
                            lldb_version_major, lldb_version_minor, lldb_version_patch
                        ),
                        line,
                    )
                )
            else:
                # Write any line that doesn't need to be converted
                output_file.write(line)


if __name__ == "__main__":
    main()
