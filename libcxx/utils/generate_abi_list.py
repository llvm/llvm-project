#!/usr/bin/env python
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import argparse
import io
import libcxx.sym_check.extract
import libcxx.sym_check.util
import pprint
import sys


def OutputFile(file):
    if isinstance(file, io.IOBase):
        return file
    assert isinstance(file, str), "Got object {} which is not a str".format(file)
    return open(file, "w", newline="\n")


def main(argv):
    parser = argparse.ArgumentParser(
        description="Extract a list of symbols from a shared library."
    )
    parser.add_argument(
        "library", metavar="LIB", type=str, help="The library to extract symbols from."
    )
    parser.add_argument(
        "-m",
        "--mapfile",
        dest="mapfile",
        default=None,
        help="The name of the mapfile that contains supplementary information about symbols. (optional, macOS-only feature)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=OutputFile,
        default=sys.stdout,
        help="The output file to write the symbols to. It is overwritten if it already exists. "
        "If no file is specified, the results are written to standard output.",
    )
    args = parser.parse_args(argv)

    symbols = libcxx.sym_check.extract.extract_symbols(args.library)
    symbols, _ = libcxx.sym_check.util.filter_stdlib_symbols(symbols)

    supplemental_info = {}
    if args.mapfile != None:
        supplemental_info = libcxx.sym_check.util.extract_object_sizes_from_map(
            args.mapfile
        )
        if len(supplemental_info) == 0:
            print("You requested that the ABI list be built with the help of a mapfile, but the specified mapfile could not be found.")
            return 1

    # Specific to the case where there is supplemental symbol information from a mapfile ...
    if len(supplemental_info) != 0:
        for sym in symbols:
            # Only update from the supplementatl information where the symbol has a
            # size, that size is not 0 and its type is OBJECT.
            if "size" not in sym or sym["size"] != 0 or sym["type"] != "OBJECT":
                continue
            if sym["name"] in supplemental_info:
                updated_size = supplemental_info[sym["name"]]
                sym["size"] = updated_size

    lines = [pprint.pformat(sym, width=99999) for sym in symbols]
    args.output.writelines("\n".join(sorted(lines)))
    return 0

if __name__ == "__main__":
    result = main(sys.argv[1:])
    sys.exit(result)
