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
    ftm.version_header_implementation,
    {
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
        ],
        "20": [
            {
                "__cpp_lib_barrier": {
                    "value": "201907L",
                    "implemented": True,
                    "need_undef": False,
                    "condition": "!defined(_LIBCPP_HAS_NO_THREADS) && _LIBCPP_AVAILABILITY_HAS_SYNC",
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
                    "implemented": True,
                    "need_undef": True,
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
                    "condition": "!defined(_LIBCPP_HAS_NO_THREADS) && _LIBCPP_AVAILABILITY_HAS_SYNC",
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
                    "implemented": True,
                    "need_undef": True,
                    "condition": None,
                },
            },
        ],
    },
)
