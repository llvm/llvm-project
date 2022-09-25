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
import re
import sys
import pathlib


@dataclass
class header:
    name: str = None
    level: int = -1


def parse_line(line: str) -> header:
    match = re.match(r"(\.+) (.+)", line)
    if not match:
        sys.exit(f"Line {line} contains invalid data.")

    # The number of periods in front of the header name is the nesting level of
    # that header.
    return header(match.group(2), len(match.group(1)))


# Generates the list of transitive includes of a header.
#
# The input contains two kinds of headers
# * Standard headers  (algorithm, string, format, etc)
# * Detail headers (__algorithm/copy.h, __algorithm/copy_n.h, etc)
# The output contains the transitive includes of the Standard header being
# processed. The detail headers are omitted from the output, but their
# transitive includes are parsed and added, if they are a Standard header.
#
# This effectively generates the dependency graph of a Standard header.
def parse_file(file: pathlib.Path) -> List[str]:
    result = list()
    with file.open(encoding="utf-8") as f:
        level = 999

        # The first line contains the Standard header being processed.
        # The transitive includes of this Standard header should be processed.
        header = parse_line(f.readline())
        assert header.level == 1
        result.append(header.name)

        for line in f.readlines():
            header = parse_line(line)

            # Skip deeper nested headers for Standard headers.
            if header.level > level:
                continue

            # Process deeper nested headers for detail headers.
            if header.name.startswith("__") or header.name.__contains__("/__"):
                level = 999
                continue

            # Add the Standard header.
            level = header.level
            result.append(header.name)

    return result


def create_include_graph(path: pathlib.Path) -> List[str]:
    result = list()
    for file in sorted(path.glob("header.*")):
        includes = parse_file(file)
        if len(includes) > 1:
            result.append(includes)
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
Typically this script is executed by libcxx/test/libcxx/transitive_includes.sh.cpp""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        default=None,
        metavar="DIR",
        help="The directory containing the transitive includes of the headers.",
    )
    options = parser.parse_args()

    root = pathlib.Path(options.input)
    print_csv(create_include_graph(root))
