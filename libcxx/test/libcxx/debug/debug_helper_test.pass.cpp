// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: windows

// Can't test the system lib because this test enables debug mode
// UNSUPPORTED: with_system_cxx_lib

// MODULES_DEFINES: _LIBCPP_DEBUG=1

#define _LIBCPP_DEBUG 1

#include <__debug>
#include "test_macros.h"
#include "debug_mode_helper.h"


template <class Func>
inline bool TestDeathTest(const char* stmt, Func&& func, DeathTest::ResultKind ExpectResult, DebugInfoMatcher Matcher = AnyMatcher) {
  DeathTest DT(Matcher);
  DeathTest::ResultKind RK = DT.Run(func);
  auto OnFailure = [&](std::string msg) {
    std::cerr << "EXPECT_DEATH( " << stmt << " ) failed! (" << msg << ")\n\n";
    if (!DT.getChildStdErr().empty()) {
      std::cerr << "---------- standard err ----------\n";
      std::cerr << DT.getChildStdErr() << "\n";
    }
    if (!DT.getChildStdOut().empty()) {
      std::cerr << "---------- standard out ----------\n";
      std::cerr << DT.getChildStdOut() << "\n";
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
  DebugInfoMatcher ExpectMatch("my message");
  TEST_DEATH_TEST_MATCHES(DeathTest::RK_MatchFailure, ExpectMatch, my_libcpp_assert());
}

void test_did_not_die() {
  TEST_DEATH_TEST(DeathTest::RK_DidNotDie, ((void)0));
}

void test_unknown() {
  TEST_DEATH_TEST(DeathTest::RK_Unknown, std::exit(13));
}

int main(int, char**)
{
  test_no_match_found();
  test_did_not_die();
  test_unknown();
  return 0;
}
