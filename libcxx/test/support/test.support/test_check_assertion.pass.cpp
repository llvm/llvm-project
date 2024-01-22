//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: availability-verbose_abort-missing

#include <cassert>
#include <cstdio>
#include <string>

#include "check_assertion.h"

template <class Func>
bool TestDeathTest(
    Outcome expected_outcome, DeathCause expected_cause, const char* stmt, Func&& func, const Matcher& matcher) {
  auto get_matcher = [&] {
#if _LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_DEBUG
    return matcher;
#else
    (void)matcher;
    return MakeAnyMatcher();
#endif
  };

  DeathTest test_case;
  DeathTestResult test_result = test_case.Run(expected_cause, func, get_matcher());
  std::string maybe_failure_description;

  Outcome outcome = test_result.outcome();
  if (expected_outcome != outcome) {
    maybe_failure_description +=
        std::string("Test outcome was different from expected; expected ") + ToString(expected_outcome) +
        ", got: " + ToString(outcome);
  }

  DeathCause cause = test_result.cause();
  if (expected_cause != cause) {
    auto failure_description =
        std::string("Cause of death was different from expected; expected ") + ToString(expected_cause) +
        ", got: " + ToString(cause);
    if (maybe_failure_description.empty()) {
      maybe_failure_description = failure_description;
    } else {
      maybe_failure_description += std::string("; ") + failure_description;
    }
  }

  if (!maybe_failure_description.empty()) {
    test_case.PrintFailureDetails(maybe_failure_description, stmt, test_result.cause());
    return false;
  }

  return true;
}

// clang-format off

#define TEST_DEATH_TEST(outcome, cause, ...)                   \
  assert(( TestDeathTest(outcome, cause, #__VA_ARGS__, [&]() { __VA_ARGS__; }, MakeAnyMatcher()) ))
#define TEST_DEATH_TEST_MATCHES(outcome, cause, matcher, ...)  \
  assert(( TestDeathTest(outcome, cause, #__VA_ARGS__, [&]() { __VA_ARGS__; }, matcher) ))

// clang-format on

#if _LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_DEBUG
DeathCause assertion_death_cause = DeathCause::VerboseAbort;
#else
DeathCause assertion_death_cause = DeathCause::Trap;
#endif

int main(int, char**) {
  auto fail_assert     = [] { _LIBCPP_ASSERT(false, "Some message"); };
  Matcher good_matcher = MakeAssertionMessageMatcher("Some message");
  Matcher bad_matcher  = MakeAssertionMessageMatcher("Bad expected message");

  // Test the implementation of death tests. We're bypassing the assertions added by the actual `EXPECT_DEATH` macros
  // which allows us to test failure cases (where the assertion would fail) as well.
  {
    // Success -- `std::terminate`.
    TEST_DEATH_TEST(Outcome::Success, DeathCause::StdTerminate, std::terminate());

    // Success -- trapping.
    TEST_DEATH_TEST(Outcome::Success, DeathCause::Trap, __builtin_trap());

    // Success -- assertion failure with any matcher.
    TEST_DEATH_TEST_MATCHES(Outcome::Success, assertion_death_cause, MakeAnyMatcher(), fail_assert());

    // Success -- assertion failure with a specific matcher.
    TEST_DEATH_TEST_MATCHES(Outcome::Success, assertion_death_cause, good_matcher, fail_assert());

#if _LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_DEBUG
    // Failure -- error message doesn't match.
    TEST_DEATH_TEST_MATCHES(Outcome::UnexpectedErrorMessage, assertion_death_cause, bad_matcher, fail_assert());
#endif

    // Invalid cause -- child did not die.
    TEST_DEATH_TEST(Outcome::InvalidCause, DeathCause::DidNotDie, ((void)0));

    // Invalid cause --  unknown.
    TEST_DEATH_TEST(Outcome::InvalidCause, DeathCause::Unknown, std::exit(13));
  }

  // Test the `EXPECT_DEATH` macros themselves. Since they assert success, we can only test successful cases.
  {
    auto invoke_abort = [] { _LIBCPP_VERBOSE_ABORT("contains some message"); };

    auto simple_matcher = [](const std::string& text) {
      bool success = text.find("some") != std::string::npos;
      return MatchResult(success, "");
    };

    EXPECT_DEATH(invoke_abort());
    EXPECT_DEATH_MATCHES(MakeAnyMatcher(), invoke_abort());
    EXPECT_DEATH_MATCHES(simple_matcher, invoke_abort());
    EXPECT_STD_TERMINATE([] { std::terminate(); });
    TEST_LIBCPP_ASSERT_FAILURE(fail_assert(), "Some message");
  }

  return 0;
}
