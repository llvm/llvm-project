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
    ftm.standard_ftms,
    {
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
        "__cpp_lib_format": {
            "c++20": "202110L",
            "c++23": "202207L",
            "c++26": "202311L",
        },
        "__cpp_lib_parallel_algorithm": {
            "c++17": "201603L",
            "c++20": "201603L",
            "c++23": "201603L",
            "c++26": "201603L",
        },
        "__cpp_lib_variant": {
            "c++17": "202102L",
            "c++20": "202106L",
            "c++23": "202106L",
            "c++26": "202306L",
        },
        "__cpp_lib_missing_FTM_in_older_standard": {
            "c++17": "2017L",
            "c++20": "2020L",
            "c++23": "2020L",
            "c++26": "2026L",
        },
    },
)
