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
        nargs=1,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to write generated header file",
        type=Path,
        required=True,
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

    [yaml_file] = args.yaml_file
    files_read = {yaml_file}

    def write_depfile():
        if not args.depfile:
            return
        deps = " ".join(str(f) for f in sorted(files_read))
        args.depfile.parent.mkdir(parents=True, exist_ok=True)
        with open(args.depfile, "w") as depfile:
            depfile.write(f"{args.output}: {deps}\n")

    header = load_yaml_file(yaml_file, HeaderFile, args.entry_point)

    if not header.template_file:
        print(f"{yaml_file}: Missing header_template", sys.stderr)
        return 2

    # The header_template path is relative to the containing YAML file.
    template_path = yaml_file.parent / header.template_file

    files_read.add(template_path)
    with open(template_path) as template:
        contents = fill_public_api(header.public_api(), template.read())

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
