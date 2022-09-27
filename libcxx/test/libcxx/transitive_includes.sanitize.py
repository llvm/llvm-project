#!/usr/bin/env python
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# This script reads lines from standard input and looks for the names of public
# C++ headers. Specifically, it looks for lines of the form 'c++/v1/header'
# where 'header' is the name of a public C++ header, excluding C compatibility
# headers. (Note it keeps the libc++ implementation detail .h files.)

# Note --trace-includes removes duplicates of included headers unless they are
# part of a cycle.

# The input looks like
# . ${build_dir}/include/c++/v1/algorithm
# .. ${build_dir}/include/c++/v1/__assert
# ... ${build_dir}/include/c++/v1/__config
# .... ${build_dir}/include/c++/v1/__config_site
# .... /usr/include/features.h
# ..... /usr/include/stdc-predef.h
# ..... /usr/include/x86_64-linux-gnu/sys/cdefs.h
# ...... /usr/include/x86_64-linux-gnu/bits/wordsize.h
# <snip>
# .... ${build_dir}/include/c++/v1/version
# .... ${build_dir}/include/c++/v1/stddef.h
# ..... /usr/lib/llvm-15/lib/clang/15.0.0/include/stddef.h
# ...... /usr/lib/llvm-15/lib/clang/15.0.0/include/__stddef_max_align_t.h
# ... ${build_dir}/include/c++/v1/type_traits
# <more>


# The filtered output will be like:
# type_traits
# . algorithm
# ... cstddef
# .... __type_traits/enable_if.h
# .... __type_traits/integral_constant.h
# .... __type_traits/is_integral.h
# ..... __type_traits/remove_cv.h
# ...... __type_traits/remove_const.h
# ...... __type_traits/remove_volatile.h
# .... version

import re
import sys

headers = []
for line in sys.stdin.readlines():
    # On Windows, the path separators can either be forward slash or backslash.
    # If it is a backslash, Clang prints it escaped as two consecutive
    # backslashes, and they need to be escaped in the RE. (Use a raw string for
    # the pattern to avoid needing another level of escaping on the Python string
    # literal level.)
    match = re.match(r"(\.+).*c\+\+(?:/|\\\\)v[0-9]+(?:/|\\\\)(.+)", line)
    if not match:
        continue

    header = match.group(2)
    # Skip C headers, but accept libc++ detail headers.
    if header.startswith("__"):
        if not header.endswith(".h"):
            continue
    elif header.endswith(".h"):
        continue

    print(f"{match.group(1)} {match.group(2)}")
