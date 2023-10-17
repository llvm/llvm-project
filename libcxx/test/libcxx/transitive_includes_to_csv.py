#!/usr/bin/env python
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from dataclasses import dataclass, field
from typing import List  # Needed for python 3.8 compatibility.
import argparse
import pathlib
import re
import sys


@dataclass
class header:
    name: str = None
    level: int = -1


def parse_line(line: str) -> header:
    """
    Parse an output line from --trace-includes into a `header`.
    """
    match = re.match(r"(\.+) (.+)", line)
    if not match:
        sys.exit(f"Line {line} contains invalid data.")

    # The number of periods in front of the header name is the nesting level of
    # that header.
    return header(match.group(2), len(match.group(1)))


# On Windows, the path separators can either be forward slash or backslash.
# If it is a backslash, Clang prints it escaped as two consecutive
# backslashes, and they need to be escaped in the RE. (Use a raw string for
# the pattern to avoid needing another level of escaping on the Python string
# literal level.)
LIBCXX_HEADER_REGEX = r".*c\+\+(?:/|\\\\)v[0-9]+(?:/|\\\\)(.+)"

def is_libcxx_header(header: str) -> bool:
    """
    Returns whether a header is a libc++ header, excluding the C-compatibility headers.
    """
    # Only keep files in the c++/vN directory.
    match = re.match(LIBCXX_HEADER_REGEX, header)
    if not match:
        return False

    # Skip C compatibility headers (in particular, make sure not to skip libc++ detail headers).
    relative = match.group(1)
    if relative.endswith(".h") and not (
        relative.startswith("__") or re.search(r"(/|\\\\)__", relative)
    ):
        return False

    return True


def parse_file(file: pathlib.Path) -> List[str]:
    """
    Parse a file containing --trace-include output to generate a list of the top-level C++ includes
    contained in it.

    This effectively generates the dependency graph of C++ Standard Library headers of the header
    whose --trace-include it is. In order to get the expected result of --trace-include, the
    -fshow-skipped-includes flag also needs to be passed.
    """
    result = list()
    with file.open(encoding="utf-8") as f:
        for line in f.readlines():
            header = parse_line(line)

            # Skip non-libc++ headers
            if not is_libcxx_header(header.name):
                continue

            # Include top-level headers in the output. There's usually exactly one,
            # except if the compiler is passed a file with `-include`. Top-level
            # headers are transparent, in the sense that we want to go look at
            # transitive includes underneath.
            if header.level == 1:
                level = 999
                result.append(header)
                continue

            # Skip libc++ headers included transitively.
            if header.level > level:
                continue

            # Detail headers are transparent too: we attribute all includes of public libc++
            # headers under a detail header to the last public libc++ header that included it.
            if header.name.startswith("__") or re.search(r"(/|\\\\)__", header.name):
                level = 999
                continue

            # Add the non-detail libc++ header to the list.
            level = header.level
            result.append(header)
    return result


def create_include_graph(trace_includes: List[pathlib.Path]) -> List[str]:
    result = list()
    for file in trace_includes:
        headers = parse_file(file)

        # Get actual filenames relative to libc++'s installation directory instead of full paths
        relative = lambda h: re.match(LIBCXX_HEADER_REGEX, h).group(1)

        top_level = relative(
            next(h.name for h in headers if h.level == 1)
        )  # There should be only one top-level header
        includes = [relative(h.name) for h in headers if h.level != 1]

        # Remove duplicates in all includes.
        includes = list(set(includes))

        if len(includes) != 0:
            result.append([top_level] + includes)
    return result


def print_csv(graph: List[str]) -> None:
    for includes in graph:
        header = includes[0]
        for include in sorted(includes[1:]):
            if header == include:
                sys.exit(f"Cycle detected: header {header} includes itself.")
            print(f"{header} {include}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Produce a dependency graph of libc++ headers, in CSV format.
This script is normally executed by libcxx/test/libcxx/transitive_includes.gen.py""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        default=None,
        metavar="FILE",
        nargs='+',
        help="One or more files containing the result of --trace-includes on the headers one wishes to graph.",
    )
    options = parser.parse_args()

    print_csv(create_include_graph(map(pathlib.Path, options.inputs)))
