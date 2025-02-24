#!/usr/bin/env python3
#
# ===- Generate headers for libc functions  ------------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==------------------------------------------------------------------------==#

import argparse
import json
import sys
from pathlib import Path

from header import HeaderFile
from yaml_to_classes import load_yaml_file, fill_public_api


def main():
    parser = argparse.ArgumentParser(description="Generate header files from YAML")
    parser.add_argument(
        "yaml_file",
        help="Path to the YAML file containing header specification",
        metavar="FILE",
        type=Path,
        nargs="+",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to write generated header file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--json",
        help="Write JSON instead of a header, can use multiple YAML files",
        action="store_true",
    )
    parser.add_argument(
        "--depfile",
        help="Path to write a depfile",
        type=Path,
    )
    parser.add_argument(
        "--write-if-changed",
        help="Write the output file only if its contents have changed",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-e",
        "--entry-point",
        help="Entry point to include; may be given many times",
        metavar="SYMBOL",
        action="append",
    )
    args = parser.parse_args()

    if not args.json and len(args.yaml_file) != 1:
        print("Only one YAML file at a time without --json", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return 2

    files_read = set()

    def write_depfile():
        if not args.depfile:
            return
        deps = " ".join(str(f) for f in sorted(files_read))
        args.depfile.parent.mkdir(parents=True, exist_ok=True)
        with open(args.depfile, "w") as depfile:
            depfile.write(f"{args.output}: {deps}\n")

    def load_yaml(path):
        files_read.add(path)
        return load_yaml_file(path, HeaderFile, args.entry_point)

    def load_header(yaml_file):
        merge_from_files = dict()

        def merge_from(paths):
            for path in paths:
                # Load each file exactly once, in case of redundant merges.
                if path in merge_from_files:
                    continue
                header = load_yaml(path)
                merge_from_files[path] = header
                merge_from(path.parent / f for f in header.merge_yaml_files)

        # Load the main file first.
        header = load_yaml(yaml_file)

        # Now load all the merge_yaml_files, and transitive merge_yaml_files.
        merge_from(yaml_file.parent / f for f in header.merge_yaml_files)

        # Merge in all those files' contents.
        for merge_from_path, merge_from_header in merge_from_files.items():
            if merge_from_header.name is not None:
                print(
                    f"{merge_from_path!s}: Merge file cannot have header field",
                    file=sys.stderr,
                )
                return 2
            header.merge(merge_from_header)

        return header

    if args.json:
        contents = json.dumps(
            [load_header(file).json_data() for file in args.yaml_file],
            indent=2,
        )
    else:
        [yaml_file] = args.yaml_file
        header = load_header(yaml_file)
        # The header_template path is relative to the containing YAML file.
        template = header.template(yaml_file.parent, files_read)
        contents = fill_public_api(header.public_api(), template)

    write_depfile()

    if (
        not args.write_if_changed
        or not args.output.exists()
        or args.output.read_text() != contents
    ):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents)


if __name__ == "__main__":
    sys.exit(main())
