# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from libcxx.test.dsl import Feature
from lit.BooleanExpression import BooleanExpression

# Helpers to define correspondances between LLVM versions and vendor system versions.
# Those are used for backdeployment features below, do not use directly in tests.
features = [
    Feature(
        name="_target-has-llvm-22",
        when=lambda cfg: BooleanExpression.evaluate(
            "TBD",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-21",
        when=lambda cfg: BooleanExpression.evaluate(
            "TBD",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-20",
        when=lambda cfg: BooleanExpression.evaluate(
            "_target-has-llvm-21 || target={{.+}}-apple-macosx{{26.[0-9](.\d+)?}}",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-19",
        when=lambda cfg: BooleanExpression.evaluate(
            "_target-has-llvm-20 || target={{.+}}-apple-macosx{{15.[4-9](.\d+)?}}",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-18",
        when=lambda cfg: BooleanExpression.evaluate(
            "_target-has-llvm-19 || target={{.+}}-apple-macosx{{15.[0-3](.\d+)?}}",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-17",
        when=lambda cfg: BooleanExpression.evaluate(
            "_target-has-llvm-18 || target={{.+}}-apple-macosx{{14.[4-9](.\d+)?}}",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-16",
        when=lambda cfg: BooleanExpression.evaluate(
            "_target-has-llvm-17 || target={{.+}}-apple-macosx{{14.[0-3](.[0-9]+)?}}",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-15",
        when=lambda cfg: BooleanExpression.evaluate(
            "_target-has-llvm-16 || target={{.+}}-apple-macosx{{13.[4-9](.[0-9]+)?}}",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-14",
        when=lambda cfg: BooleanExpression.evaluate(
            "_target-has-llvm-15",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-13",
        when=lambda cfg: BooleanExpression.evaluate(
            "_target-has-llvm-14 || target={{.+}}-apple-macosx{{13.[0-3](.[0-9]+)?}}",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-12",
        when=lambda cfg: BooleanExpression.evaluate(
            "_target-has-llvm-13 || target={{.+}}-apple-macosx{{12.[3-9](.[0-9]+)?}}",
            cfg.available_features,
        ),
    ),
]

# Define features for back-deployment testing.
#
# These features can be used to XFAIL tests that fail when deployed on (or compiled
# for) an older system. For example, if a test exhibits a bug in the libc++ on a
# particular system version, or if it uses a symbol that is not available on an
# older version of the dylib, it can be marked as XFAIL with these features.
#
# We have two families of Lit features:
#
# The first one is `using-built-library-before-llvm-XYZ`. These features encode the
# fact that the test suite is being *run* against a version of the shared/static library
# that predates LLVM version XYZ. This is useful to represent the use case of compiling
# a program against the latest libc++ but then deploying it and running it on an older
# system with an older version of the (usually shared) library.
#
# This feature is built up using the target triple passed to the compiler and the
# `stdlib=system` Lit feature, which encodes that we're running against the same library
# as described by the target triple.
#
# The second set of features is `availability-<FEATURE>-missing`. This family of Lit
# features encodes the presence of availability markup in the libc++ headers. This is
# useful to check that a test fails specifically when compiled for a given deployment
# target, such as when testing availability markup where we want to make sure that
# using the annotated facility on a deployment target that doesn't support it will fail
# at compile time. This can be achieved by creating a `.verify.cpp` test that checks for
# the right errors and marking the test as `REQUIRES: availability-<FEATURE>-missing`.
#
# This feature is built up using the presence of availability markup detected inside
# __config, the flavor of the library being tested and the target triple passed to the
# compiler.
#
# Note that both families of Lit features are similar but different in important ways.
# For example, tests for availability markup should be expected to produce diagnostics
# regardless of whether we're running against a system library, as long as we're using
# a libc++ flavor that enables availability markup. Similarly, a test could fail when
# run against the system library of an older version of FreeBSD, even though FreeBSD
# doesn't provide availability markup at the time of writing this.
for version in ("12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"):
    features.append(
        Feature(
            name="using-built-library-before-llvm-{}".format(version),
            when=lambda cfg, v=version: BooleanExpression.evaluate(
                "stdlib=system && !_target-has-llvm-{}".format(v),
                cfg.available_features,
            ),
        )
    )

features += [
    # Tests that require https://wg21.link/P0482 support in the built library
    Feature(
        name="availability-char8_t_support-missing",
        when=lambda cfg: BooleanExpression.evaluate(
            "!libcpp-has-no-availability-markup && (stdlib=apple-libc++ && !_target-has-llvm-12)",
            cfg.available_features,
        ),
    ),
    # Tests that require std::to_chars(floating-point) in the built library
    Feature(
        name="availability-fp_to_chars-missing",
        when=lambda cfg: BooleanExpression.evaluate(
            "!libcpp-has-no-availability-markup && (stdlib=apple-libc++ && !_target-has-llvm-14)",
            cfg.available_features,
        ),
    ),
    # Tests that require __libcpp_verbose_abort support in the built library
    Feature(
        name="availability-verbose_abort-missing",
        when=lambda cfg: BooleanExpression.evaluate(
            "!libcpp-has-no-availability-markup && (stdlib=apple-libc++ && !_target-has-llvm-15)",
            cfg.available_features,
        ),
    ),
    # Tests that require std::pmr support in the built library
    Feature(
        name="availability-pmr-missing",
        when=lambda cfg: BooleanExpression.evaluate(
            "!libcpp-has-no-availability-markup && (stdlib=apple-libc++ && !_target-has-llvm-16)",
            cfg.available_features,
        ),
    ),
    # Tests that require support for <print> and std::print in <ostream> in the built library.
    Feature(
        name="availability-print-missing",
        when=lambda cfg: BooleanExpression.evaluate(
            "!libcpp-has-no-availability-markup && (stdlib=apple-libc++ && !_target-has-llvm-18)",
            cfg.available_features,
        ),
    ),
    # Tests that require time zone database support in the built library
    Feature(
        name="availability-tzdb-missing",
        when=lambda cfg: BooleanExpression.evaluate(
            "!libcpp-has-no-availability-markup && (stdlib=apple-libc++ && !_target-has-llvm-19)",
            cfg.available_features,
        ),
    ),
    # Tests that require std::from_chars(floating-point) in the built library
    Feature(
        name="availability-fp_from_chars-missing",
        when=lambda cfg: BooleanExpression.evaluate(
            "!libcpp-has-no-availability-markup && (stdlib=apple-libc++ && !_target-has-llvm-20)",
            cfg.available_features,
        ),
    ),
]
