# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# RUN: %{python} %s %{libcxx-dir}/utils %{libcxx-dir}/test/libcxx/feature_test_macro/test_data.json

import sys

sys.path.append(sys.argv[1])
from generate_feature_test_macro_components import FeatureTestMacros


def test(output, expected):
    assert output == expected, f"expected\n{expected}\n\noutput\n{output}"


ftm = FeatureTestMacros(sys.argv[2])
test(
    ftm.version_header,
    """// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_VERSION
#define _LIBCPP_VERSION

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 17
#  define __cpp_lib_any 201606L
#  define __cpp_lib_parallel_algorithm 201603L
#  define __cpp_lib_variant 202102L
// define __cpp_lib_missing_FTM_in_older_standard 2017L
#endif // _LIBCPP_STD_VER >= 17

#if _LIBCPP_STD_VER >= 20
#  if !defined(_LIBCPP_HAS_NO_THREADS) && _LIBCPP_AVAILABILITY_HAS_SYNC
#    define __cpp_lib_barrier 201907L
#  endif
// define __cpp_lib_format 202110L
// define __cpp_lib_variant 202106L
// define __cpp_lib_missing_FTM_in_older_standard 2020L
#endif // _LIBCPP_STD_VER >= 20

#if _LIBCPP_STD_VER >= 23
// define __cpp_lib_format 202207L
#endif // _LIBCPP_STD_VER >= 23

#if _LIBCPP_STD_VER >= 26
#  if !defined(_LIBCPP_HAS_NO_THREADS) && _LIBCPP_AVAILABILITY_HAS_SYNC
#    undef __cpp_lib_barrier
#    define __cpp_lib_barrier 299900L
#  endif
// define __cpp_lib_format 202311L
// define __cpp_lib_variant 202306L
// define __cpp_lib_missing_FTM_in_older_standard 2026L
#endif // _LIBCPP_STD_VER >= 26

#endif // _LIBCPP_VERSION
""",
)
