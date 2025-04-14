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

    def test_implementation(self):

        expected = {
            "__cpp_lib_any": {
                "c++17": "201606L",
                "c++20": "201606L",
                "c++23": "201606L",
                "c++26": "201606L",
            },
            "__cpp_lib_barrier": {
                "c++20": "201907L",
                "c++23": "201907L",
                "c++26": "299900L",
            },
            "__cpp_lib_format": {},
            "__cpp_lib_parallel_algorithm": {
                "c++17": "201603L",
                "c++20": "201603L",
                "c++23": "201603L",
                "c++26": "201603L",
            },
            "__cpp_lib_variant": {
                "c++17": "202102L",
                "c++20": "202102L",
                "c++23": "202102L",
                "c++26": "202102L",
            },
            "__cpp_lib_missing_FTM_in_older_standard": {},
        }

        self.assertEqual(self.ftm.implemented_ftms, expected)


if __name__ == "__main__":
    unittest.main()
