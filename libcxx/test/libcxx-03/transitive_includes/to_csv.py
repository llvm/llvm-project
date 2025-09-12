#!/usr/bin/env python
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from typing import List, Tuple, Optional
import argparse
import io
import itertools
import os
import pathlib
import re
import sys

libcxx_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(libcxx_root, "utils"))
from libcxx.header_information import Header

def parse_line(line: str) -> Tuple[int, str]:
    """
    Parse a single line of --trace-includes output.

    Returns the inclusion level and the raw file name being included.
    """
    match = re.match(r"(\.+) (.+)", line)
    if not match:
        raise ArgumentError(f"Line {line} contains invalid data.")

    # The number of periods in front of the header name is the nesting level of
    # that header.
    return (len(match.group(1)), match.group(2))

def make_cxx_v1_relative(header: str) -> Optional[str]:
    """
    Returns the path of the header as relative to <whatever>/c++/v1, or None if the path
    doesn't contain c++/v1.

    We use that heuristic to figure out which headers are libc++ headers.
    """
    # On Windows, the path separators can either be forward slash or backslash.
    # If it is a backslash, Clang prints it escaped as two consecutive
    # backslashes, and they need to be escaped in the RE. (Use a raw string for
    # the pattern to avoid needing another level of escaping on the Python string
    # literal level.)
    pathsep = r"(?:/|\\\\)"
    CXX_V1_REGEX = r"^.*c\+\+" + pathsep + r"v[0-9]+" + pathsep + r"(.+)$"
    match = re.match(CXX_V1_REGEX, header)
    if not match:
        return None
    else:
        return match.group(1)

def parse_file(file: io.TextIOBase) -> List[Tuple[Header, Header]]:
    """
    Parse a file containing --trace-includes output to generate a list of the
    transitive includes contained in it.
    """
    result = []
    includer = None
    for line in file.readlines():
        (level, header) = parse_line(line)
        relative = make_cxx_v1_relative(header)

        # Not a libc++ header
        if relative is None:
            continue

        # If we're at the first level, remember this header as being the one who includes other headers.
        # There's usually exactly one, except if the compiler is passed a file with `-include`.
        if level == 1:
            includer = Header(relative)
            continue

        # Otherwise, take note that this header is being included by the top-level includer.
        else:
            assert includer is not None
            result.append((includer, Header(relative)))
    return result

def print_csv(includes: List[Tuple[Header, Header]]) -> None:
    """
    Print the transitive includes as space-delimited CSV.

    This function only prints public libc++ headers that are not C compatibility headers.
    """
    # Sort and group by includer
    by_includer = lambda t: t[0]
    includes = itertools.groupby(sorted(includes, key=by_includer), key=by_includer)

    for (includer, includees) in includes:
        includees = map(lambda t: t[1], includees)
        for h in sorted(set(includees)):
            if h.is_public() and not h.is_C_compatibility():
                print(f"{includer} {h}")

def main(argv):
    parser = argparse.ArgumentParser(
        description="""
        Given a list of headers produced by --trace-includes, produce a list of libc++ headers in that output.

        Note that -fshow-skipped-includes must also be passed to the compiler in order to get sufficient
        information for this script to run.

        The output of this script is provided in space-delimited CSV format where each line contains:

            <header performing inclusion> <header being included>
        """)
    parser.add_argument("inputs", type=argparse.FileType("r"), nargs='+', default=None,
        help="One or more files containing the result of --trace-includes")
    args = parser.parse_args(argv)

    includes = [line for file in args.inputs for line in parse_file(file)]
    print_csv(includes)

if __name__ == "__main__":
    main(sys.argv[1:])
