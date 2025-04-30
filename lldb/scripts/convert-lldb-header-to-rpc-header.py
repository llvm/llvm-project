#!/usr/bin/env python3
"""
Usage: convert-lldb-header-to-rpc-header.py <path/to/input-header.h> <path/to/output-header.h>

This scripts takes common LLDB headers (such as lldb-defines.h) and replaces references to LLDB
with those for RPC. This happens for:
- namespace definitions
- namespace usage
- version string macros
- ifdef/ifndef lines
"""

import argparse
import os
import re


INCLUDES_TO_REMOVE_REGEX = re.compile(
    r'#include "lldb/lldb-forward|#include "lldb/lldb-versioning'
)
LLDB_GUARD_REGEX = re.compile(r".+LLDB_LLDB_")
LLDB_API_GUARD_REGEX = re.compile(r".+LLDB_API_")
LLDB_VERSION_REGEX = re.compile(r"#define LLDB_VERSION")
LLDB_REVISION_REGEX = re.compile(r"#define LLDB_REVISION")
LLDB_VERSION_STRING_REGEX = re.compile(r"#define LLDB_VERSION_STRING")
LLDB_LOCAL_INCLUDE_REGEX = re.compile(r'#include "lldb/lldb-')
LLDB_NAMESPACE_DEFINITION_REGEX = re.compile(r".*namespace lldb")
LLDB_NAMESPACE_REGEX = re.compile(r".+lldb::")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    input_path = str(args.input)
    output_path = str(args.output)
    with open(input_path, "r") as input_file:
        lines = input_file.readlines()

    with open(output_path, "w") as output_file:
        for line in lines:
            # NOTE: We do not use lldb-forward.h or lldb-versioning.h in RPC, so remove
            # all includes that are found for these files.
            if INCLUDES_TO_REMOVE_REGEX.match(line):
                continue
            # For lldb-rpc-defines.h, replace the ifndef LLDB_LLDB_ portion with LLDB_RPC_ as we're not
            # using LLDB private definitions in RPC.
            elif LLDB_GUARD_REGEX.match(line):
                output_file.write(re.sub(r"LLDB_LLDB_", r"LLDB_RPC_", line))
            # Similarly to lldb-rpc-defines.h, replace the ifndef for LLDB_API in SBDefines.h to LLDB_RPC_API_ for the same reason.
            elif LLDB_API_GUARD_REGEX.match(line):
                output_file.write(re.sub(r"LLDB_API_", r"LLDB_RPC_API_", line))
            # Replace the references for the macros that define the versioning strings in
            # lldb-rpc-defines.h.
            # NOTE: Here we assume that the versioning info has already been uncommented and
            # populated from the original lldb-defines.h.
            elif LLDB_VERSION_REGEX.match(line):
                output_file.write(re.sub(r"LLDB_VERSION", r"LLDB_RPC_VERSION", line))
            elif LLDB_REVISION_REGEX.match(line):
                output_file.write(re.sub(r"LLDB_REVISION", r"LLDB_RPC_REVISION", line))
            elif LLDB_VERSION_STRING_REGEX.match(line):
                output_file.write(
                    re.sub(r"LLDB_VERSION_STRING", r"LLDB_RPC_VERSION_STRING", line)
                )
            # For local #includes
            elif LLDB_LOCAL_INCLUDE_REGEX.match(line):
                output_file.write(re.sub(r"lldb/lldb-", r"lldb-rpc-", line))
            # Rename the lldb namespace definition to lldb-rpc.
            elif LLDB_NAMESPACE_DEFINITION_REGEX.match(line):
                output_file.write(re.sub(r"lldb", r"lldb_rpc", line))
            # Rename namespace references
            elif LLDB_NAMESPACE_REGEX.match(line):
                output_file.write(re.sub(r"lldb::", r"lldb_rpc::", line))
            else:
                # Write any line that doesn't need to be converted
                output_file.write(line)


if __name__ == "__main__":
    main()
