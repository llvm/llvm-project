#!/usr/bin/env python
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
"""
sym_diff - Compare two symbol lists and output the differences.
"""

from argparse import ArgumentParser
import sys
from libcxx.sym_check import diff, util


def main():
    parser = ArgumentParser(
        description="Extract a list of symbols from a shared library."
    )
    parser.add_argument(
        "--names-only",
        dest="names_only",
        help="Only print symbol names",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--removed-only",
        dest="removed_only",
        help="Only print removed symbols",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--only-stdlib-symbols",
        dest="only_stdlib",
        help="Filter all symbols not related to the stdlib",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        help="Exit with a non-zero status if any symbols " "differ",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="The output file. stdout is used if not given",
        type=str,
        action="store",
        default=None,
    )
    parser.add_argument(
        "--demangle", dest="demangle", action="store_true", default=False
    )
    parser.add_argument(
        "--mapfile-old_syms",
        dest="mapfile_old_syms",
        help="The name of the mapfile that contains supplementary information about old symbols. (optional, macOS-only feature)",
        type=str,
        action="store",
        default=None,
    )
    parser.add_argument(
        "--mapfile-new_syms",
        dest="mapfile_new_syms",
        help="The name of the mapfile that contains supplementary information about new symbols. (optional, macOS-only feature)",
        type=str,
        action="store",
        default=None,
    )
    parser.add_argument(
        "old_syms",
        metavar="old-syms",
        type=str,
        help="The file containing the old symbol list or a library",
    )
    parser.add_argument(
        "new_syms",
        metavar="new-syms",
        type=str,
        help="The file containing the new symbol list or a library",
    )

    args = parser.parse_args()

    old_syms_supplemental_info = {}
    new_syms_supplemental_info = {}
    if args.mapfile_old_syms != None:
        (
            map_extract_success,
            old_syms_supplemental_info,
        ) = util.extract_object_sizes_from_map(args.mapfile_old_syms)
        if not map_extract_success:
            print(
                f"ERROR: Request to check the ABI with the help of a mapfile, but the specified mapfile ({args.mapfile}) could not be found."
            )
            return 1

    if args.mapfile_new_syms != None:
        (
            map_extract_success,
            new_syms_supplemental_info,
        ) = util.extract_object_sizes_from_map(args.mapfile_new_syms)
        if not map_extract_success:
            print(
                f"ERROR: Request to check the ABI with the help of a mapfile, but the specified mapfile ({args.mapfile}) could not be found."
            )
            return 1

    old_syms_list = util.extract_or_load(args.old_syms)
    # If there is a mapfile, update the symbols with its supplemental information.
    if len(old_syms_supplemental_info) != 0:
        util.update_symbols_with_supplemental_information(
            old_syms_list, old_syms_supplemental_info
        )

    new_syms_list = util.extract_or_load(args.new_syms)
    # If there is a mapfile, update the symbols with its supplemental information.
    if len(new_syms_supplemental_info) != 0:
        util.update_symbols_with_supplemental_information(
            new_syms_list, new_syms_supplemental_info
        )

    if args.only_stdlib:
        old_syms_list, _ = util.filter_stdlib_symbols(old_syms_list)
        new_syms_list, _ = util.filter_stdlib_symbols(new_syms_list)

    added, removed, changed = diff.diff(old_syms_list, new_syms_list)
    if args.removed_only:
        added = {}
    report, is_break, is_different = diff.report_diff(
        added, removed, changed, names_only=args.names_only, demangle=args.demangle
    )
    if args.output is None:
        print(report)
    else:
        with open(args.output, "w") as f:
            f.write(report + "\n")
    exit_code = 1 if is_break or (args.strict and is_different) else 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
