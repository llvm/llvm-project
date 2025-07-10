#!/usr/bin/env python3
"""
Usage: -i <path/to/input-header.h> -o <path/to/output-header.h> -m LLDB_MAJOR_VERSION -n LLDB_MINOR_VERSION -p LLDB_PATCH_VERSION

This script uncomments and populates the versioning information in lldb-defines.h. Note that the LLDB version numbering looks like MAJOR.MINOR.PATCH
"""

import argparse
import os
import re

LLDB_VERSION_REGEX = re.compile(r"//\s*#define LLDB_VERSION\s*$", re.M)
LLDB_REVISION_REGEX = re.compile(r"//\s*#define LLDB_REVISION\s*$", re.M)
LLDB_VERSION_STRING_REGEX = re.compile(r"//\s*#define LLDB_VERSION_STRING\s*$", re.M)


def main():
    parser = argparse.ArgumentParser(
        description="This script uncomments and populates the versioning information in lldb-defines.h. Note that the LLDB version numbering looks like MAJOR.MINOR.PATCH"
    )
    parser.add_argument("-i", "--input_path", help="The filepath for the input header.")
    parser.add_argument(
        "-o", "--output_path", help="The filepath for the output header."
    )
    parser.add_argument("-m", "--major", help="The LLDB version major.")
    parser.add_argument("-n", "--minor", help="The LLDB version minor.")
    parser.add_argument("-p", "--patch", help="The LLDB version patch number.")
    args = parser.parse_args()
    input_path = str(args.input_path)
    output_path = str(args.output_path)

    with open(input_path, "r") as input_file:
        lines = input_file.readlines()
        file_buffer = "".join(lines)

    with open(output_path, "w") as output_file:
        # For the defines in lldb-defines.h that define the major, minor and version string
        # uncomment each define and populate its value using the arguments passed in.
        # e.g. //#define LLDB_VERSION -> #define LLDB_VERSION <LLDB_MAJOR_VERSION>
        file_buffer = re.sub(
            LLDB_VERSION_REGEX,
            r"#define LLDB_VERSION " + args.major,
            file_buffer,
        )

        file_buffer = re.sub(
            LLDB_REVISION_REGEX,
            r"#define LLDB_REVISION " + args.patch,
            file_buffer,
        )
        file_buffer = re.sub(
            LLDB_VERSION_STRING_REGEX,
            r'#define LLDB_VERSION_STRING "{0}.{1}.{2}"'.format(
                args.major, args.minor, args.patch
            ),
            file_buffer,
        )
        output_file.write(file_buffer)


if __name__ == "__main__":
    main()
