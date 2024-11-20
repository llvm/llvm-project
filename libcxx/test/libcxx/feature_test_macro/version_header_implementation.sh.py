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
        self.maxDiff = None # This causes the diff to be printed when the test fails

    def test_implementation(self):
        expected = {
            "17": [
                {
                    "__cpp_lib_any": {
                        "value": "201606L",
                        "implemented": True,
                        "need_undef": False,
                        "condition": None,
                    },
                },
                {
                    "__cpp_lib_parallel_algorithm": {
                        "value": "201603L",
                        "implemented": True,
                        "need_undef": False,
                        "condition": None,
                    },
                },
                {
                    "__cpp_lib_variant": {
                        "value": "202102L",
                        "implemented": True,
                        "need_undef": False,
                        "condition": None,
                    },
                },
                {
                    "__cpp_lib_missing_FTM_in_older_standard": {
                        "value": "2017L",
                        "implemented": False,
                        "need_undef": False,
                        "condition": None,
                    },
                },
            ],
            "20": [
                {
                    "__cpp_lib_barrier": {
                        "value": "201907L",
                        "implemented": True,
                        "need_undef": False,
                        "condition": "_LIBCPP_HAS_THREADS && _LIBCPP_AVAILABILITY_HAS_SYNC",
                    },
                },
                {
                    "__cpp_lib_format": {
                        "value": "202110L",
                        "implemented": False,
                        "need_undef": False,
                        "condition": None,
                    },
                },
                {
                    "__cpp_lib_variant": {
                        "value": "202106L",
                        "implemented": False,
                        "need_undef": False,
                        "condition": None,
                    },
                },
                {
                    "__cpp_lib_missing_FTM_in_older_standard": {
                        "value": "2020L",
                        "implemented": False,
                        "need_undef": False,
                        "condition": None,
                    },
                },
            ],
            "23": [
                {
                    "__cpp_lib_format": {
                        "value": "202207L",
                        "implemented": False,
                        "need_undef": False,
                        "condition": None,
                    },
                },
            ],
            "26": [
                {
                    "__cpp_lib_barrier": {
                        "value": "299900L",
                        "implemented": True,
                        "need_undef": True,
                        "condition": "_LIBCPP_HAS_THREADS && _LIBCPP_AVAILABILITY_HAS_SYNC",
                    },
                },
                {
                    "__cpp_lib_format": {
                        "value": "202311L",
                        "implemented": False,
                        "need_undef": False,
                        "condition": None,
                    },
                },
                {
                    "__cpp_lib_variant": {
                        "value": "202306L",
                        "implemented": False,
                        "need_undef": False,
                        "condition": None,
                    },
                },
                {
                    "__cpp_lib_missing_FTM_in_older_standard": {
                        "value": "2026L",
                        "implemented": False,
                        "need_undef": False,
                        "condition": None,
                    },
                },
            ],
        }

        self.assertEqual(self.ftm.version_header_implementation, expected)

if __name__ == '__main__':
    unittest.main()
