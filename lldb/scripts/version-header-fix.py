#!/usr/bin/env python3
"""
Usage: <path/to/input-header.h> <path/to/output-header.h> LLDB_MAJOR_VERSION LLDB_MINOR_VERSION LLDB_PATCH_VERSION

This script uncomments and populates the versioning information in lldb-defines.h
"""

import argparse
import os
import re

LLDB_VERSION_REGEX = re.compile(r"//\s*#define LLDB_VERSION\s*$", re.M)
LLDB_REVISION_REGEX = re.compile(r"//\s*#define LLDB_REVISION\s*$", re.M)
LLDB_VERSION_STRING_REGEX = re.compile(r"//\s*#define LLDB_VERSION_STRING\s*$", re.M)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("lldb_version_major")
    parser.add_argument("lldb_version_minor")
    parser.add_argument("lldb_version_patch")
    args = parser.parse_args()
    input_path = str(args.input_path)
    output_path = str(args.output_path)
    lldb_version_major = args.lldb_version_major
    lldb_version_minor = args.lldb_version_minor
    lldb_version_patch = args.lldb_version_patch

    with open(input_path, "r") as input_file:
        lines = input_file.readlines()
        file_buffer = "".join(lines)

    with open(output_path, "w") as output_file:
        # For the defines in lldb-defines.h that define the major, minor and version string
        # uncomment each define and populate its value using the arguments passed in.
        # e.g. //#define LLDB_VERSION -> #define LLDB_VERSION <LLDB_MAJOR_VERSION>
        file_buffer = re.sub(
            LLDB_VERSION_REGEX,
            r"#define LLDB_VERSION " + lldb_version_major,
            file_buffer,
        )

        file_buffer = re.sub(
            LLDB_REVISION_REGEX,
            r"#define LLDB_REVISION " + lldb_version_patch,
            file_buffer,
        )
        file_buffer = re.sub(
            LLDB_VERSION_STRING_REGEX,
            r'#define LLDB_VERSION_STRING "{0}.{1}.{2}"'.format(
                lldb_version_major, lldb_version_minor, lldb_version_patch
            ),
            file_buffer,
        )
        output_file.write(file_buffer)


if __name__ == "__main__":
    main()
