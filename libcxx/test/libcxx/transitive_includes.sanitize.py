#!/usr/bin/env python
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# This script reads lines from standard input and looks for the names of public C++ headers.
# Specifically, it looks for lines of the form 'c++/v1/header' where 'header' is the name
# of a public C++ header, excluding C compatibility headers.

# The input looks like
#. ${build_dir}/include/c++/v1/algorithm
#.. ${build_dir}/include/c++/v1/__assert
#... ${build_dir}/include/c++/v1/__config
#.... ${build_dir}/include/c++/v1/__config_site
#.... /usr/include/features.h
#..... /usr/include/stdc-predef.h
#..... /usr/include/x86_64-linux-gnu/sys/cdefs.h
#...... /usr/include/x86_64-linux-gnu/bits/wordsize.h
# <snip>
#.... ${build_dir}/include/c++/v1/version
#.... ${build_dir}/include/c++/v1/stddef.h
#..... /usr/lib/llvm-15/lib/clang/15.0.0/include/stddef.h
#...... /usr/lib/llvm-15/lib/clang/15.0.0/include/__stddef_max_align_t.h
#... ${build_dir}/include/c++/v1/type_traits
# <more>

# The first line matched libc++ header contains the name of the header being
# evaluated. The might be other headers before, for example ASAN adds
# additional headers. The filtered output will be like:
# version
# type_traits

import os
import re
import sys

# Determine the top-level header in the input.
top_level_header = None
while True:
    line = sys.stdin.readline()
    # On Windows, the path separators can either be forward slash or backslash.
    # If it is a backslash, Clang prints it escaped as two consecutive
    # backslashes, and they need to be escaped in the RE. (Use a raw string for
    # the pattern to avoid needing another level of escaping on the Python string
    # literal level.)
    match = re.match(
        r". .*(?:/|\\\\)include(?:/|\\\\)c\+\+(?:/|\\\\)v[0-9]+(?:/|\\\\)(.+)", line
    )
    if match:
        top_level_header = match.group(1)
        break

# Filter out non Standard transitive includes.
headers = []
for line in sys.stdin.readlines():
    match = re.search(r"c\+\+(?:/|\\\\)v[0-9]+(?:/|\\\\)(.+)", line)
    if not match:
        continue

    header = match.group(1)
    if os.path.basename(header).endswith(".h"):  # Skip C headers
        continue

    if os.path.basename(header).startswith("__"):  # Skip internal headers
        continue

    if header == top_level_header:
        sys.exit(f"Cyclic dependency in header {header}")

    headers.append(header)

print("\n".join(sorted(set(headers))))
