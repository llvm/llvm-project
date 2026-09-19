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
        name="_target-has-llvm-23",
        when=lambda cfg: BooleanExpression.evaluate(
            "TBD",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-22",
        when=lambda cfg: BooleanExpression.evaluate(
            r"_target-has-llvm-23 || target={{.+}}-apple-macosx{{27.[0-9](.\d+)?}}",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-21",
        when=lambda cfg: BooleanExpression.evaluate(
            r"_target-has-llvm-22 || target={{.+}}-apple-macosx{{26.[4-9](.\d+)?}}",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-20",
        when=lambda cfg: BooleanExpression.evaluate(
            r"_target-has-llvm-21 || target={{.+}}-apple-macosx{{26.[0-3](.\d+)?}}",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-19",
        when=lambda cfg: BooleanExpression.evaluate(
            r"_target-has-llvm-20 || target={{.+}}-apple-macosx{{15.[4-9](.\d+)?}}",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-18",
        when=lambda cfg: BooleanExpression.evaluate(
            r"_target-has-llvm-19 || target={{.+}}-apple-macosx{{15.[0-3](.\d+)?}}",
            cfg.available_features,
        ),
    ),
    Feature(
        name="_target-has-llvm-17",
        when=lambda cfg: BooleanExpression.evaluate(
            r"_target-has-llvm-18 || target={{.+}}-apple-macosx{{14.[4-9](.\d+)?}}",
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
for version in ("12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"):
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
    # Tests that require std::text_encoding::environment() in the built library
    Feature(
        name="availability-te-environment-missing",
        when=lambda cfg: BooleanExpression.evaluate(
            "!libcpp-has-no-availability-markup && (stdlib=apple-libc++ && !_target-has-llvm-23)",
            cfg.available_features,
        ),
    ),
    # Tests that require std::stacktrace in the built library
    Feature(
        name="availability-stacktrace-missing",
        when=lambda cfg: BooleanExpression.evaluate(
            "(!libcpp-has-no-availability-markup && (stdlib=apple-libc++ && !_target-has-llvm-23))"
            # 32-bit x86 Android's own (non-LLVM) unwinder is unreliable before API 24: it's the
            # same legacy i686-linux-android(21|22|23) combination that llvm-libc++-android.cfg.in
            # already works around a separate stack-misalignment bug for (see the -mstackrealign
            # comment there and https://github.com/android/ndk/issues/693). Here it manifests as
            # std::stacktrace::current() silently capturing an empty/truncated trace rather than
            # a crash, so just treat stacktrace as unavailable on this narrow legacy combination.
            "|| (target={{i686-linux-android.*}} && android-device-api={{2[123]}})"
            # 32-bit x86 Windows: capture crashes outright (RtlCaptureContext's frame-pointer
            # capture is unreliable on i686) rather than just producing a bad trace. Not treated
            # as worth chasing further; declared unsupported here rather than fixed.
            "|| target={{i686-.*windows.*}}",
            cfg.available_features,
        ),
    ),
    # Tests that require std::stacktrace_entry::source_file()/description() to resolve to
    # something non-empty for a real capture. On bare-metal targets (no dynamic loader, no
    # filesystem), there's no dl_iterate_phdr/proc-self-exe equivalent to learn the running
    # image's own name from, so entries can never be associated with a source file/description,
    # even though capturing a stacktrace itself works fine.
    Feature(
        name="availability-stacktrace-no-image-info",
        when=lambda cfg: BooleanExpression.evaluate(
            "target={{.*-none-eabi.*}}",
            cfg.available_features,
        ),
    ),
]
