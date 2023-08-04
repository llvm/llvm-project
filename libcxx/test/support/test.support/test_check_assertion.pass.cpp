//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=unchecked
// XFAIL: availability-verbose_abort-missing

#include <cassert>
#include <cstdio>
#include <string>

#include "check_assertion.h"

template <class Func>
inline bool TestDeathTest(const char* stmt, Func&& func, DeathTest::ResultKind ExpectResult, AssertionInfoMatcher Matcher = AnyMatcher) {
  DeathTest DT(Matcher);
  DeathTest::ResultKind RK = DT.Run(func);
  auto OnFailure = [&](std::string msg) {
    std::fprintf(stderr, "EXPECT_DEATH( %s ) failed! (%s)\n\n", stmt, msg.c_str());
    if (!DT.getChildStdErr().empty()) {
      std::fprintf(stderr, "---------- standard err ----------\n%s\n", DT.getChildStdErr().c_str());
    }
    if (!DT.getChildStdOut().empty()) {
      std::fprintf(stderr, "---------- standard out ----------\n%s\n", DT.getChildStdOut().c_str());
    }
    return false;
  };
  if (RK != ExpectResult)
    return OnFailure(std::string("expected result did not occur: expected ") + DeathTest::ResultKindToString(ExpectResult) + " got: " + DeathTest::ResultKindToString(RK));
  return true;
}
#define TEST_DEATH_TEST(RK, ...) assert((TestDeathTest(#__VA_ARGS__, [&]() { __VA_ARGS__; }, RK, AnyMatcher )))

#define TEST_DEATH_TEST_MATCHES(RK, Matcher, ...) assert((TestDeathTest(#__VA_ARGS__, [&]() { __VA_ARGS__; }, RK, Matcher)))

void my_libcpp_assert() {
  _LIBCPP_ASSERT(false, "other");
}

void test_no_match_found() {
  AssertionInfoMatcher ExpectMatch("my message");
  TEST_DEATH_TEST_MATCHES(DeathTest::RK_MatchFailure, ExpectMatch, my_libcpp_assert());
}

void test_did_not_die() {
  TEST_DEATH_TEST(DeathTest::RK_DidNotDie, ((void)0));
}

void test_unknown() {
  TEST_DEATH_TEST(DeathTest::RK_Unknown, std::exit(13));
}

int main(int, char**) {
  test_no_match_found();
  test_did_not_die();
  test_unknown();
  return 0;
}
