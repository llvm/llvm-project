# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# Make sure that libc++ headers work when defining _XOPEN_SOURCE=500.
# We may not want to guarantee this forever, but since this works today and
# it's something that users rely on, it makes sense to put a test on it.
#
# https://github.com/llvm/llvm-project/issues/117630

# RUN: %{python} %s %{libcxx-dir}/utils

import sys

sys.path.append(sys.argv[1])
from libcxx.header_information import (
    lit_header_restrictions,
    lit_header_undeprecations,
    public_headers,
)

for header in public_headers:
    for version in (500, 600, 700):
        # TODO: <fstream> currently uses ::fseeko unguarded, which fails with _XOPEN_SOURCE=500.
        if header == "fstream" and version == 500:
            continue

        print(
            f"""\
//--- {header}.xopen_source_{version}.compile.pass.cpp
{lit_header_restrictions.get(header, '')}
{lit_header_undeprecations.get(header, '')}

// ADDITIONAL_COMPILE_FLAGS: -D_XOPEN_SOURCE={version}

#include <{header}>
"""
        )
