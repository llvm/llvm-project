#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Test that we don't remove transitive includes of public C++ headers in the library accidentally.
# When we remove a transitive public include, clients tend to break because they don't always
# properly include what they use. Note that we don't check which system (C) headers are
# included transitively, because that is too unstable across platforms, and hence difficult
# to test for.
#
# This is not meant to block libc++ from removing unused transitive includes
# forever, however we do try to group removals for a couple of releases
# to avoid breaking users at every release.

# RUN: %{python} %s %{libcxx-dir}/utils

# block Lit from interpreting a RUN/XFAIL/etc inside the generation script
# END.

import sys
sys.path.append(sys.argv[1])
from libcxx.header_information import lit_header_restrictions, public_headers

import re

# To re-generate the list of expected headers, temporarily set this to True, and run this test.
# Note that this needs to be done for all supported language versions of libc++:
# for std in c++03 c++11 c++14 c++17 c++20 c++23 c++26; do <build>/bin/llvm-lit --param std=$std libcxx/test/libcxx/transitive_includes.gen.py; done
regenerate_expected_results = False

if regenerate_expected_results:
    print(
        f"""\
//--- generate-transitive-includes.sh.cpp
// RUN: mkdir %t
"""
    )

    all_traces = []
    for header in sorted(public_headers):
        if header.endswith(".h"):  # Skip C compatibility or detail headers
            continue

        normalized_header = re.sub("/", "_", header)
        print(
            f"""\
// RUN: echo "#include <{header}>" | %{{cxx}} -xc++ - %{{flags}} %{{compile_flags}} --trace-includes -fshow-skipped-includes --preprocess > /dev/null 2> %t/trace-includes.{normalized_header}.txt
"""
        )
        all_traces.append(f"%t/trace-includes.{normalized_header}.txt")

    print(
        f"""\
// RUN: %{{python}} %{{libcxx-dir}}/test/libcxx/transitive_includes_to_csv.py {' '.join(all_traces)} > %{{libcxx-dir}}/test/libcxx/transitive_includes/%{{cxx_std}}.csv
"""
    )

else:
    for header in public_headers:
        if header.endswith(".h"):  # Skip C compatibility or detail headers
            continue

        # Escape slashes for the awk command below
        escaped_header = header.replace("/", "\\/")

        print(
            f"""\
//--- {header}.sh.cpp
{lit_header_restrictions.get(header, '')}

// TODO: Fix this test to make it work with localization or wide characters disabled
// UNSUPPORTED: no-localization, no-wide-characters, no-threads, no-filesystem, libcpp-has-no-experimental-tzdb, no-tzdb

// When built with modules, this test doesn't work because --trace-includes doesn't
// report the stack of includes correctly.
// UNSUPPORTED: clang-modules-build

// This test uses --trace-includes, which is not supported by GCC.
// UNSUPPORTED: gcc

// This test is not supported when we remove the transitive includes provided for backwards
// compatibility. When we bulk-remove them, we'll adjust the includes that are expected by
// this test instead.
// UNSUPPORTED: transitive-includes-disabled

// TODO: Figure out why <stdatomic.h> doesn't work on FreeBSD
// UNSUPPORTED: LIBCXX-FREEBSD-FIXME

// RUN: mkdir %t
// RUN: %{{cxx}} %s %{{flags}} %{{compile_flags}} --trace-includes -fshow-skipped-includes --preprocess > /dev/null 2> %t/trace-includes.txt
// RUN: %{{python}} %{{libcxx-dir}}/test/libcxx/transitive_includes_to_csv.py %t/trace-includes.txt > %t/actual_transitive_includes.csv
// RUN: cat %{{libcxx-dir}}/test/libcxx/transitive_includes/%{{cxx_std}}.csv | awk '/^{escaped_header} / {{ print }}' > %t/expected_transitive_includes.csv
// RUN: diff -w %t/expected_transitive_includes.csv %t/actual_transitive_includes.csv
#include <{header}>
"""
        )
