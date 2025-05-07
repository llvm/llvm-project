# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# RUN: %{python} %s %{libcxx-dir}/utils %{libcxx-dir}/test/libcxx/feature_test_macro/test_data.json

import sys
import unittest

UTILS = sys.argv[1]
TEST_DATA = sys.argv[2]
del sys.argv[1:3]

sys.path.append(UTILS)
from generate_feature_test_macro_components import FeatureTestMacros


class Test(unittest.TestCase):
    def setUp(self):
        self.ftm = FeatureTestMacros(TEST_DATA)
        self.maxDiff = None  # This causes the diff to be printed when the test fails

    def test_implementeation(self):
        expected = """// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_VERSIONH
#define _LIBCPP_VERSIONH

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
#  if _LIBCPP_HAS_THREADS && _LIBCPP_AVAILABILITY_HAS_SYNC
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
#  if _LIBCPP_HAS_THREADS && _LIBCPP_AVAILABILITY_HAS_SYNC
#    undef __cpp_lib_barrier
#    define __cpp_lib_barrier 299900L
#  endif
// define __cpp_lib_format 202311L
// define __cpp_lib_variant 202306L
// define __cpp_lib_missing_FTM_in_older_standard 2026L
#endif // _LIBCPP_STD_VER >= 26

#endif // _LIBCPP_VERSIONH
"""
        self.assertEqual(self.ftm.version_header, expected)


if __name__ == "__main__":
    unittest.main()
