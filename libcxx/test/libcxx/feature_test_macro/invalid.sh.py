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
from generate_feature_test_macro_components import FeatureTestMacros, DataNotSorted


def test(output, expected):
    assert output == expected, f"expected\n{expected}\n\noutput\n{output}"


def test_error(data, type, message):
    tmp = sys.argv[2]
    with open(tmp, "w") as file:
        file.write(json.dumps(data))
    ftm = FeatureTestMacros(tmp, ["charconv"])
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

test_error(
    [
        {
            "name": "abc",
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
        {
            "name": "ghi",
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
        { # This entry is in the wrong alphabetic order
            "name": "def",
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
        {
            "name": "jkl",
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
    DataNotSorted,
    """\
The ftm names are not sorted.
--- input data
+++ sorted data
@@ -1,4 +1,4 @@
 abc
+def
 ghi
-def
 jkl
""",
)

test_error(
    [
        {
            "name": "abc",
            "values": {
                "c++14": {
                    "197001": [
                        {
                            "implemented": False,
                        },
                    ],
                },
                "c++23": {
                    "197001": [
                        {
                            "implemented": False,
                        },
                    ],
                },
                # This entry is in the wrong alphabetic order
                # Note we don't use C++98, but C++03 instead so alphabetic order
                # works this century.
                "c++20": {
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
    DataNotSorted,
    """\
The C++ standard version numbers of ftm 'abc' are not sorted.
--- input data
+++ sorted data
@@ -1,3 +1,3 @@
 c++14
+c++20
 c++23
-c++20
""",
)

test_error(
    [
        {
            "name": "abc",
            "values": {
                "c++14": {
                    "197001": [
                        {
                            "implemented": False,
                        },
                    ],
                    "197002": [
                        {
                            "implemented": False,
                        },
                    ],
                    "197004": [
                        {
                            "implemented": False,
                        },
                    ],
                    # This entry is in the wrong alphabetic order
                    "197003": [
                        {
                            "implemented": False,
                        },
                    ],
                    "197005": [
                        {
                            "implemented": False,
                        },
                    ],
                },
            },
            "headers": [],
        },
    ],
    DataNotSorted,
    """\
The value of the fmt 'abc' in c++14 are not sorted.
--- input data
+++ sorted data
@@ -1,5 +1,5 @@
 197001
 197002
+197003
 197004
-197003
 197005
""",
)
