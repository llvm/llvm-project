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
    r'#include "lldb/lldb-forward.h"|#include "lldb/lldb-versioning.h"'
)
LLDB_GUARD_REGEX = re.compile(r"(?P<guard_type>#.+)LLDB_LLDB_\s*", re.M)
LLDB_API_GUARD_REGEX = re.compile(r"(?P<guard_type>#.+)LLDB_API_\s*", re.M)
LLDB_VERSION_REGEX = re.compile(r"#define LLDB_VERSION", re.M)
LLDB_REVISION_REGEX = re.compile(r"#define LLDB_REVISION", re.M)
LLDB_VERSION_STRING_REGEX = re.compile(r"#define LLDB_VERSION_STRING", re.M)
LLDB_LOCAL_INCLUDE_REGEX = re.compile(r'#include "lldb/lldb-\s*', re.M)
LLDB_NAMESPACE_DEFINITION_REGEX = re.compile(
    r"(?P<comment_marker>//\s*){,1}namespace lldb\s{1}", re.M
)
LLDB_NAMESPACE_REGEX = re.compile(r"\s*.+lldb::\s*", re.M)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    input_path = str(args.input)
    output_path = str(args.output)
    with open(input_path, "r") as input_file:
        lines = input_file.readlines()
        file_buffer = "".join(lines)

    with open(output_path, "w") as output_file:
        # NOTE: We do not use lldb-forward.h or lldb-versioning.h in RPC, so remove
        # all includes that are found for these files.
        file_buffer = re.sub(INCLUDES_TO_REMOVE_REGEX, r"", file_buffer)

        # For lldb-rpc-defines.h, replace the ifndef LLDB_LLDB_ portion with LLDB_RPC_ as we're not
        # using LLDB private definitions in RPC.
        lldb_guard_matches = LLDB_GUARD_REGEX.finditer(file_buffer)
        for match in lldb_guard_matches:
            file_buffer = re.sub(
                match.group(),
                r"{0}LLDB_RPC_".format(match.group("guard_type")),
                file_buffer,
            )

        # Similarly to lldb-rpc-defines.h, replace the ifndef for LLDB_API in SBDefines.h to LLDB_RPC_API_ for the same reason.
        lldb_api_guard_matches = LLDB_API_GUARD_REGEX.finditer(file_buffer)
        for match in lldb_api_guard_matches:
            file_buffer = re.sub(
                match.group(),
                r"{0}LLDB_RPC_API_".format(match.group("guard_type")),
                file_buffer,
            )

        # Replace the references for the macros that define the versioning strings in
        # lldb-rpc-defines.h.
        # NOTE: Here we assume that the versioning info has already been uncommented and
        # populated from the original lldb-defines.h.
        file_buffer = re.sub(
            LLDB_VERSION_REGEX, r"#define LLDB_RPC_VERSION", file_buffer
        )
        file_buffer = re.sub(
            LLDB_REVISION_REGEX, r"#define LLDB_RPC_REVISION", file_buffer
        )
        file_buffer = re.sub(
            LLDB_VERSION_STRING_REGEX, r"#define LLDB_RPC_VERSION_STRING", file_buffer
        )

        # For local #includes
        file_buffer = re.sub(
            LLDB_LOCAL_INCLUDE_REGEX, r'#include "lldb-rpc-', file_buffer
        )

        # Rename the lldb namespace definition to lldb-rpc.
        lldb_rpc_namespace_definition_matches = (
            LLDB_NAMESPACE_DEFINITION_REGEX.finditer(file_buffer)
        )
        for match in lldb_rpc_namespace_definition_matches:
            comment_marker = (
                match.group("comment_marker") if match.group("comment_marker") else ""
            )
            file_buffer = re.sub(
                match.group(),
                r"{0}namespace lldb_rpc ".format(comment_marker),
                file_buffer,
            )

        # Rename the lldb namespace definition to lldb-rpc.
        file_buffer = re.sub(LLDB_NAMESPACE_REGEX, r"lldb_rpc::", file_buffer)

        output_file.write(file_buffer)


if __name__ == "__main__":
    main()
