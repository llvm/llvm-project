# RUN: %{python} %s %{libcxx-dir}/utils %{libcxx-dir}/utils/data/feature_test_macros/test_data.json
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import sys

sys.path.append(sys.argv[1])
from generate_feature_test_macro_components import feature_test_macros


def test(output, expected):
    assert output == expected, f"expected\n{expected}\n\noutput\n{output}"


fmt = feature_test_macros(sys.argv[2])
test(
    fmt.get_std_dialects(),
    [
        "c++17",
        "c++20",
        "c++23",
        "c++26",
    ],
)
