//===-- Implementation of the base class for libc unittests----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcTest.h"

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/fixed_point/fx_rep.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h" // LIBC_TYPES_HAS_INT128
#include "src/__support/uint128.h"
#include "test/UnitTest/TestLogger.h"

#if __STDC_HOSTED__
#include <time.h>
#define LIBC_TEST_USE_CLOCK
#elif defined(TARGET_SUPPORTS_CLOCK)
#include <time.h>

#include "src/time/clock.h"
extern "C" clock_t clock() noexcept { return LIBC_NAMESPACE::clock(); }
#define LIBC_TEST_USE_CLOCK
#endif

namespace LIBC_NAMESPACE_DECL {
namespace testing {

namespace internal {

TestLogger &operator<<(TestLogger &logger, Location Loc) {
  return logger << Loc.file << ":" << Loc.line << ": FAILURE\n";
}

// When the value is UInt128, __uint128_t or wider, show its hexadecimal
// digits.
template <typename T>
cpp::enable_if_t<(cpp::is_integral_v<T> && (sizeof(T) > sizeof(uint64_t))) ||
                     is_big_int_v<T>,
                 cpp::string>
describeValue(T Value) {
  const IntegerToString<T, radix::Hex::WithPrefix> buffer(Value);
  return buffer.view();
}

// When the value is of a standard integral type, just display it as normal.
template <typename T>
cpp::enable_if_t<cpp::is_integral_v<T> && (sizeof(T) <= sizeof(uint64_t)),
                 cpp::string>
describeValue(T Value) {
  return cpp::to_string(Value);
}

#ifdef LIBC_COMPILER_HAS_FIXED_POINT
template <typename T>
cpp::enable_if_t<cpp::is_fixed_point_v<T>, cpp::string> describeValue(T Value) {
  using FXRep = fixed_point::FXRep<T>;
  using comp_t = typename FXRep::CompType;

  return cpp::to_string(cpp::bit_cast<comp_t>(Value)) + " * 2^-" +
         cpp::to_string(FXRep::FRACTION_LEN);
}
#endif // LIBC_COMPILER_HAS_FIXED_POINT

cpp::string_view describeValue(const cpp::string &Value) { return Value; }
cpp::string_view describeValue(cpp::string_view Value) { return Value; }

template <typename ValType>
bool test(RunContext *Ctx, TestCond Cond, ValType LHS, ValType RHS,
          const char *LHSStr, const char *RHSStr, Location Loc) {
  auto ExplainDifference = [=, &Ctx](bool Cond,
                                     cpp::string_view OpString) -> bool {
    if (Cond)
      return true;
    Ctx->markFail();
    size_t OffsetLength = OpString.size() > 2 ? OpString.size() - 2 : 0;
    cpp::string Offset(OffsetLength, ' ');
    tlog << Loc;
    tlog << Offset << "Expected: " << LHSStr << '\n'
         << Offset << "Which is: " << describeValue(LHS) << '\n'
         << "To be " << OpString << ": " << RHSStr << '\n'
         << Offset << "Which is: " << describeValue(RHS) << '\n';
    return false;
  };

  switch (Cond) {
  case TestCond::EQ:
    return ExplainDifference(LHS == RHS, "equal to");
  case TestCond::NE:
    return ExplainDifference(LHS != RHS, "not equal to");
  case TestCond::LT:
    return ExplainDifference(LHS < RHS, "less than");
  case TestCond::LE:
    return ExplainDifference(LHS <= RHS, "less than or equal to");
  case TestCond::GT:
    return ExplainDifference(LHS > RHS, "greater than");
  case TestCond::GE:
    return ExplainDifference(LHS >= RHS, "greater than or equal to");
  }
  __builtin_unreachable();
}

} // namespace internal

Test *Test::Start = nullptr;
Test *Test::End = nullptr;

int argc = 0;
char **argv = nullptr;
char **envp = nullptr;

using internal::RunContext;

void Test::addTest(Test *T) {
  if (End == nullptr) {
    Start = T;
    End = T;
    return;
  }

  End->Next = T;
  End = T;
}

int Test::getNumTests() {
  int N = 0;
  for (Test *T = Start; T; T = T->Next, ++N)
    ;
  return N;
}

int Test::runTests(const TestOptions &Options) {
  const char *green = Options.PrintColor ? "\033[32m" : "";
  const char *red = Options.PrintColor ? "\033[31m" : "";
  const char *reset = Options.PrintColor ? "\033[0m" : "";

  int TestCount = getNumTests();
  if (TestCount) {
    tlog << green << "[==========] " << reset << "Running " << TestCount
         << " test";
    if (TestCount > 1)
      tlog << "s";
    tlog << " from 1 test suite.\n";
  }

  int FailCount = 0;
  for (Test *T = Start; T != nullptr; T = T->Next) {
    const char *TestName = T->getName();

    if (Options.TestFilter && cpp::string(TestName) != Options.TestFilter) {
      --TestCount;
      continue;
    }

    tlog << green << "[ RUN      ] " << reset << TestName << '\n';
    [[maybe_unused]] const uint64_t start_time = clock();
    RunContext Ctx;
    T->SetUp();
    T->setContext(&Ctx);
    T->Run();
    T->TearDown();
    [[maybe_unused]] const uint64_t end_time = clock();
    switch (Ctx.status()) {
    case RunContext::RunResult::Fail:
      tlog << red << "[  FAILED  ] " << reset << TestName << '\n';
      ++FailCount;
      break;
    case RunContext::RunResult::Pass:
      tlog << green << "[       OK ] " << reset << TestName;
#ifdef LIBC_TEST_USE_CLOCK
      tlog << " (";
      if (start_time > end_time) {
        tlog << "unknown - try rerunning)\n";
      } else {
        const auto duration = end_time - start_time;
        const uint64_t duration_ms = (duration * 1000) / CLOCKS_PER_SEC;
        const uint64_t duration_us = (duration * 1000 * 1000) / CLOCKS_PER_SEC;
        const uint64_t duration_ns =
            (duration * 1000 * 1000 * 1000) / CLOCKS_PER_SEC;
        if (Options.TimeInMs || duration_ms != 0)
          tlog << duration_ms << " ms)\n";
        else if (duration_us != 0)
          tlog << duration_us << " us)\n";
        else
          tlog << duration_ns << " ns)\n";
      }
#else
      tlog << '\n';
#endif
      break;
    }
  }

  if (TestCount > 0) {
    tlog << "Ran " << TestCount << " tests. "
         << " PASS: " << TestCount - FailCount << ' ' << " FAIL: " << FailCount
         << '\n';
  } else {
    tlog << "No tests run.\n";
    if (Options.TestFilter) {
      tlog << "No matching test for " << Options.TestFilter << '\n';
    }
  }

  return FailCount > 0 || TestCount == 0 ? 1 : 0;
}

namespace internal {

#define TEST_SPECIALIZATION(TYPE)                                              \
  template bool test<TYPE>(RunContext * Ctx, TestCond Cond, TYPE LHS,          \
                           TYPE RHS, const char *LHSStr, const char *RHSStr,   \
                           Location Loc)

TEST_SPECIALIZATION(char);
TEST_SPECIALIZATION(short);
TEST_SPECIALIZATION(int);
TEST_SPECIALIZATION(long);
TEST_SPECIALIZATION(long long);

TEST_SPECIALIZATION(unsigned char);
TEST_SPECIALIZATION(unsigned short);
TEST_SPECIALIZATION(unsigned int);
TEST_SPECIALIZATION(unsigned long);
TEST_SPECIALIZATION(unsigned long long);

TEST_SPECIALIZATION(bool);

// We cannot just use a single UInt128 specialization as that resolves to only
// one type, UInt<128> or __uint128_t. We want both overloads as we want to
#ifdef LIBC_TYPES_HAS_INT128
// When builtin __uint128_t type is available, include its specialization
// also.
TEST_SPECIALIZATION(__uint128_t);
#endif // LIBC_TYPES_HAS_INT128

TEST_SPECIALIZATION(LIBC_NAMESPACE::Int<128>);

TEST_SPECIALIZATION(LIBC_NAMESPACE::UInt<96>);
TEST_SPECIALIZATION(LIBC_NAMESPACE::UInt<128>);
TEST_SPECIALIZATION(LIBC_NAMESPACE::UInt<192>);
TEST_SPECIALIZATION(LIBC_NAMESPACE::UInt<256>);
TEST_SPECIALIZATION(LIBC_NAMESPACE::UInt<320>);

TEST_SPECIALIZATION(LIBC_NAMESPACE::cpp::string_view);
TEST_SPECIALIZATION(LIBC_NAMESPACE::cpp::string);

#ifdef LIBC_COMPILER_HAS_FIXED_POINT
TEST_SPECIALIZATION(short fract);
TEST_SPECIALIZATION(fract);
TEST_SPECIALIZATION(long fract);
TEST_SPECIALIZATION(unsigned short fract);
TEST_SPECIALIZATION(unsigned fract);
TEST_SPECIALIZATION(unsigned long fract);

TEST_SPECIALIZATION(short accum);
TEST_SPECIALIZATION(accum);
TEST_SPECIALIZATION(long accum);
TEST_SPECIALIZATION(unsigned short accum);
TEST_SPECIALIZATION(unsigned accum);
TEST_SPECIALIZATION(unsigned long accum);
#endif // LIBC_COMPILER_HAS_FIXED_POINT

} // namespace internal

bool Test::testStrEq(const char *LHS, const char *RHS, const char *LHSStr,
                     const char *RHSStr, internal::Location Loc) {
  return internal::test(
      Ctx, TestCond::EQ, LHS ? cpp::string_view(LHS) : cpp::string_view(),
      RHS ? cpp::string_view(RHS) : cpp::string_view(), LHSStr, RHSStr, Loc);
}

bool Test::testStrNe(const char *LHS, const char *RHS, const char *LHSStr,
                     const char *RHSStr, internal::Location Loc) {
  return internal::test(
      Ctx, TestCond::NE, LHS ? cpp::string_view(LHS) : cpp::string_view(),
      RHS ? cpp::string_view(RHS) : cpp::string_view(), LHSStr, RHSStr, Loc);
}

bool Test::testMatch(bool MatchResult, MatcherBase &Matcher, const char *LHSStr,
                     const char *RHSStr, internal::Location Loc) {
  if (MatchResult)
    return true;

  Ctx->markFail();
  if (!Matcher.is_silent()) {
    tlog << Loc;
    tlog << "Failed to match " << LHSStr << " against " << RHSStr << ".\n";
    Matcher.explainError();
  }
  return false;
}

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL
