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
        self.ftm = FeatureTestMacros(TEST_DATA)
        self.maxDiff = None  # This causes the diff to be printed when the test fails

    def test_implementation(self):
        expected = {
            "__cpp_lib_any": Metadata(
                headers=["any"], test_suite_guard=None, libcxx_guard=None
            ),
            "__cpp_lib_barrier": Metadata(
                headers=["barrier"],
                test_suite_guard="!defined(_LIBCPP_VERSION) || (_LIBCPP_HAS_THREADS && _LIBCPP_AVAILABILITY_HAS_SYNC)",
                libcxx_guard="_LIBCPP_HAS_THREADS && _LIBCPP_AVAILABILITY_HAS_SYNC",
            ),
            "__cpp_lib_format": Metadata(
                headers=["format"], test_suite_guard=None, libcxx_guard=None
            ),
            "__cpp_lib_parallel_algorithm": Metadata(
                headers=["algorithm", "numeric"],
                test_suite_guard=None,
                libcxx_guard=None,
            ),
            "__cpp_lib_variant": Metadata(
                headers=["variant"], test_suite_guard=None, libcxx_guard=None
            ),
            "__cpp_lib_missing_FTM_in_older_standard": Metadata(
                headers=[], test_suite_guard=None, libcxx_guard=None
            ),
        }
        self.assertEqual(self.ftm.ftm_metadata, expected)


if __name__ == "__main__":
    unittest.main()
