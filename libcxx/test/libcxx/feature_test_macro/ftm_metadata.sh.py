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
from generate_feature_test_macro_components import FeatureTestMacros, Metadata


def test(output, expected):
    assert output == expected, f"expected\n{expected}\n\noutput\n{output}"


ftm = FeatureTestMacros(sys.argv[2])

test(
    ftm.ftm_metadata,
    {
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
            headers=["algorithm", "numeric"], test_suite_guard=None, libcxx_guard=None
        ),
        "__cpp_lib_variant": Metadata(
            headers=["variant"], test_suite_guard=None, libcxx_guard=None
        ),
        "__cpp_lib_missing_FTM_in_older_standard": Metadata(
            headers=[], test_suite_guard=None, libcxx_guard=None
        ),
    },
)
