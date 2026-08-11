# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from libcxx.test.dsl import Feature, programSucceeds
from lit.BooleanExpression import BooleanExpression

features = []

# Detect the hardening mode that the tests are being compiled with as `libcpp-hardening-mode=<mode>`.
#
# Note that this is the mode in effect when compiling the tests, which is not necessarily the mode
# the library was configured with, since it can be overridden with compiler flags.
for mode in ("none", "fast", "extensive", "debug"):
    check_program = f"""
        #include <stddef.h> // any header to get the definitions
        int main(int, char**) {{
        #if defined(_LIBCPP_VERSION) && \\
                defined(_LIBCPP_HARDENING_MODE) && _LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_{mode.upper()}
            return 0;
        #else
            return 1;
        #endif
        }}
    """
    features.append(
        Feature(
            name=f"libcpp-hardening-mode={mode}",
            when=lambda cfg, prog=check_program: programSucceeds(cfg, prog)
        )
    )

# Detect the assertion semantic used by hardening as `libcpp-assertion-semantic=<semantic>`.
#
# Like the hardening mode above, this is the semantic in effect when compiling the tests, whether it
# comes from how the library was configured or from elsewhere.
for semantic in ("ignore", "observe", "quick_enforce", "enforce"):
    check_program = f"""
        #include <stddef.h> // any header to get the definitions
        int main(int, char**) {{
        #if defined(_LIBCPP_VERSION) && \\
                defined(_LIBCPP_ASSERTION_SEMANTIC) && \\
                _LIBCPP_ASSERTION_SEMANTIC == _LIBCPP_ASSERTION_SEMANTIC_{semantic.upper()}
            return 0;
        #else
            return 1;
        #endif
        }}
    """
    features.append(
        Feature(
            name=f"libcpp-assertion-semantic={semantic}",
            when=lambda cfg, prog=check_program: programSucceeds(cfg, prog)
        )
    )

# Whether the test suite is able to check that hardening assertions fire.
#
# Such tests are written using the `TEST_LIBCPP_ASSERT_FAILURE` macro from `check_assertion.h`,
# which runs the code that should trigger the assertion in a child process and inspects how that
# process died. That machinery requires Unix headers (it uses `fork` and pipes), C++11 or later,
# and localization support (it uses `<regex>` and `<sstream>`).
#
# On top of that, a failing assertion must be observable at all, which depends on the assertion
# semantic in effect:
# - with `ignore`, the assertion isn't even evaluated, so there is nothing to observe;
# - with `enforce`, the failure is reported through `std::__libcpp_verbose_abort`. When that function
#   isn't available in the library we're running against, `_LIBCPP_VERBOSE_ABORT` degrades to a bare
#   `abort()`, so neither the assertion message nor the way the process died match what the test suite
#   expects.
_can_test_hardening_assertions = " && ".join(
    [
        "stdlib=libc++",
        "has-unix-headers",
        "!c++03",
        "!no-localization",
        "!libcpp-assertion-semantic=ignore",
        "!(libcpp-assertion-semantic=enforce && availability-verbose_abort-missing)",
    ]
)

features.append(
    Feature(
        name="can-test-hardening-assertions",
        when=lambda cfg: BooleanExpression.evaluate(
            _can_test_hardening_assertions, cfg.available_features
        ),
    )
)

# Whether the test suite is able to check that hardening assertions of a given category fire.
#
# Assertion categories are enabled by different hardening modes, so the suffix is the weakest mode in
# which the assertion under test is enabled. For example `_LIBCPP_ASSERT_NON_NULL` is enabled in the
# `extensive` and `debug` modes, so a test for such an assertion should use
# `can-test-hardening-assertions-extensive`.
enabling_modes = {
    "fast": ("fast", "extensive", "debug"),
    "extensive": ("extensive", "debug"),
    "debug": ("debug",),
}
for category, modes in enabling_modes.items():
    expression = "can-test-hardening-assertions && ({})".format(" || ".join(f"libcpp-hardening-mode={mode}" for mode in modes))
    features.append(
        Feature(
            name=f"can-test-hardening-assertions-{category}",
            when=lambda cfg, expr=expression: BooleanExpression.evaluate(expr, cfg.available_features)
        )
    )
