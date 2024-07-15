# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# RUN: %{python} %s %{libcxx-dir}/utils %t

import sys
import json

sys.path.append(sys.argv[1])
from generate_feature_test_macro_components import FeatureTestMacros


def test(output, expected):
    assert output == expected, f"expected\n{expected}\n\noutput\n{output}"


def test_error(data, type, message):
    tmp = sys.argv[2]
    with open(tmp, "w") as file:
        file.write(json.dumps(data))
    ftm = FeatureTestMacros(tmp)
    try:
        ftm.implemented_ftms
    except type as error:
        test(str(error), message)
    else:
        assert False, "no exception was thrown"


test_error(
    [
        {
            "values": {
                "c++17": {
                    "197001": [
                        {
                            "implemented": False,
                        },
                    ],
                },
            },
            "headers": [],
        },
    ],
    KeyError,
    "'name'",
)

test_error(
    [
        {
            "name": "a",
            "headers": [],
        },
    ],
    KeyError,
    "'values'",
)

test_error(
    [
        {
            "name": "a",
            "values": {},
            "headers": [],
        },
    ],
    AssertionError,
    "'values' is empty",
)


test_error(
    [
        {
            "name": "a",
            "values": {
                "c++17": {},
            },
            "headers": [],
        },
    ],
    AssertionError,
    "a[c++17] has no entries",
)

test_error(
    [
        {
            "name": "a",
            "values": {
                "c++17": {
                    "197001": [
                        {},
                    ],
                },
            },
            "headers": [],
        },
    ],
    KeyError,
    "'implemented'",
)
