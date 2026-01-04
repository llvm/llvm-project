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
from generate_feature_test_macro_components import FeatureTestMacros, Metadata


class Test(unittest.TestCase):
    def setUp(self):
        self.ftm = FeatureTestMacros(TEST_DATA, ["charconv"])
        self.maxDiff = None  # This causes the diff to be printed when the test fails

    def test_implementation(self):
        # FTM not available in C++14.
        self.assertEqual(self.ftm.is_implemented("__cpp_lib_any", "c++14"), False)
        self.assertEqual(self.ftm.is_implemented("__cpp_lib_any", "c++17"), True)

        self.assertEqual(self.ftm.is_implemented("__cpp_lib_format", "c++20"), False)

        # FTM C++20 202106L, libc++ has 202102L
        self.assertEqual(self.ftm.is_implemented("__cpp_lib_variant", "c++20"), False)


if __name__ == "__main__":
    unittest.main()
