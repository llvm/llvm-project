# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from libcxx.test.dsl import Feature, testMacros

features = []

# Features describing parts of the library that can be carved out, like localization
# or threading support.
#
# These are detected by asking <test_macros.h> whether it considers the carve-out to be
# in effect. This keeps the logic for determining whether a carve-out applies in a single
# place, and it guarantees that these Lit features agree with what the tests use.
carveouts = {
    "TEST_HAS_NO_FILESYSTEM": "no-filesystem",
    "TEST_HAS_NO_LOCALIZATION": "no-localization",
    "TEST_HAS_NO_RANDOM_DEVICE": "no-random-device",
    "TEST_HAS_NO_THREADS": "no-threads",
    "TEST_HAS_NO_TIME_ZONE_DATABASE": "no-tzdb",
    "TEST_HAS_NO_UNICODE": "libcpp-has-no-unicode",
    "TEST_HAS_NO_WIDE_CHARACTERS": "no-wide-characters",
}
for macro, feature in carveouts.items():
    features.append(
        Feature(
            name=feature,
            when=lambda cfg, m=macro: m in testMacros(cfg),
        )
    )
