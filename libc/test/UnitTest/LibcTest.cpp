//===-- Implementation of the base class for libc unittests----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcTest.h"

#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/UInt128.h"
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

namespace LIBC_NAMESPACE {
namespace testing {

namespace internal {

TestLogger &operator<<(TestLogger &logger, Location Loc) {
  return logger << Loc.file << ":" << Loc.line << ": FAILURE\n";
}

// When the value is UInt128, __uint128_t or wider, show its hexadecimal
// digits.
template <typename T>
cpp::enable_if_t<cpp::is_integral_v<T> && (sizeof(T) > sizeof(uint64_t)),
                 cpp::string>
describeValue(T Value) {
  static_assert(sizeof(T) % 8 == 0, "Unsupported size of UInt");
  const IntegerToString<T, radix::Hex::WithPrefix> buffer(Value);
  return buffer.view();
}

// When the value is of a standard integral type, just display it as normal.
template <typename ValType>
cpp::enable_if_t<cpp::is_integral_v<ValType> &&
                     sizeof(ValType) <= sizeof(uint64_t),
                 cpp::string>
describeValue(ValType Value) {
  return cpp::to_string(Value);
}

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

int Test::runTests(const char *TestFilter) {
  int TestCount = 0;
  int FailCount = 0;
  for (Test *T = Start; T != nullptr; T = T->Next) {
    const char *TestName = T->getName();
    cpp::string StrTestName(TestName);
    constexpr auto GREEN = "\033[32m";
    constexpr auto RED = "\033[31m";
    constexpr auto RESET = "\033[0m";
    if ((TestFilter != nullptr) && (StrTestName != TestFilter)) {
      continue;
    }
    tlog << GREEN << "[ RUN      ] " << RESET << TestName << '\n';
    [[maybe_unused]] const auto start_time = clock();
    RunContext Ctx;
    T->SetUp();
    T->setContext(&Ctx);
    T->Run();
    T->TearDown();
    [[maybe_unused]] const auto end_time = clock();
    switch (Ctx.status()) {
    case RunContext::RunResult::Fail:
      tlog << RED << "[  FAILED  ] " << RESET << TestName << '\n';
      ++FailCount;
      break;
    case RunContext::RunResult::Pass:
      tlog << GREEN << "[       OK ] " << RESET << TestName;
#ifdef LIBC_TEST_USE_CLOCK
      tlog << " (took ";
      if (start_time > end_time) {
        tlog << "unknown - try rerunning)\n";
      } else {
        const auto duration = end_time - start_time;
        const uint64_t duration_ms = (duration * 1000) / CLOCKS_PER_SEC;
        const uint64_t duration_us = (duration * 1000 * 1000) / CLOCKS_PER_SEC;
        const uint64_t duration_ns =
            (duration * 1000 * 1000 * 1000) / CLOCKS_PER_SEC;
        if (duration_ms != 0)
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
    ++TestCount;
  }

  if (TestCount > 0) {
    tlog << "Ran " << TestCount << " tests. "
         << " PASS: " << TestCount - FailCount << ' ' << " FAIL: " << FailCount
         << '\n';
  } else {
    tlog << "No tests run.\n";
    if (TestFilter) {
      tlog << "No matching test for " << TestFilter << '\n';
    }
  }

  return FailCount > 0 || TestCount == 0 ? 1 : 0;
}

namespace internal {

template bool test<char>(RunContext *Ctx, TestCond Cond, char LHS, char RHS,
                         const char *LHSStr, const char *RHSStr, Location Loc);

template bool test<short>(RunContext *Ctx, TestCond Cond, short LHS, short RHS,
                          const char *LHSStr, const char *RHSStr, Location Loc);

template bool test<int>(RunContext *Ctx, TestCond Cond, int LHS, int RHS,
                        const char *LHSStr, const char *RHSStr, Location Loc);

template bool test<long>(RunContext *Ctx, TestCond Cond, long LHS, long RHS,
                         const char *LHSStr, const char *RHSStr, Location Loc);

template bool test<long long>(RunContext *Ctx, TestCond Cond, long long LHS,
                              long long RHS, const char *LHSStr,
                              const char *RHSStr, Location Loc);

template bool test<unsigned char>(RunContext *Ctx, TestCond Cond,
                                  unsigned char LHS, unsigned char RHS,
                                  const char *LHSStr, const char *RHSStr,
                                  Location Loc);

template bool test<unsigned short>(RunContext *Ctx, TestCond Cond,
                                   unsigned short LHS, unsigned short RHS,
                                   const char *LHSStr, const char *RHSStr,
                                   Location Loc);

template bool test<unsigned int>(RunContext *Ctx, TestCond Cond,
                                 unsigned int LHS, unsigned int RHS,
                                 const char *LHSStr, const char *RHSStr,
                                 Location Loc);

template bool test<unsigned long>(RunContext *Ctx, TestCond Cond,
                                  unsigned long LHS, unsigned long RHS,
                                  const char *LHSStr, const char *RHSStr,
                                  Location Loc);

template bool test<bool>(RunContext *Ctx, TestCond Cond, bool LHS, bool RHS,
                         const char *LHSStr, const char *RHSStr, Location Loc);

template bool test<unsigned long long>(RunContext *Ctx, TestCond Cond,
                                       unsigned long long LHS,
                                       unsigned long long RHS,
                                       const char *LHSStr, const char *RHSStr,
                                       Location Loc);

// We cannot just use a single UInt128 specialization as that resolves to only
// one type, UInt<128> or __uint128_t. We want both overloads as we want to
// be able to unittest UInt<128> on platforms where UInt128 resolves to
// UInt128.
#ifdef __SIZEOF_INT128__
// When builtin __uint128_t type is available, include its specialization
// also.
template bool test<__uint128_t>(RunContext *Ctx, TestCond Cond, __uint128_t LHS,
                                __uint128_t RHS, const char *LHSStr,
                                const char *RHSStr, Location Loc);
#endif

template bool test<LIBC_NAMESPACE::cpp::Int<128>>(
    RunContext *Ctx, TestCond Cond, LIBC_NAMESPACE::cpp::Int<128> LHS,
    LIBC_NAMESPACE::cpp::Int<128> RHS, const char *LHSStr, const char *RHSStr,
    Location Loc);

template bool test<LIBC_NAMESPACE::cpp::UInt<128>>(
    RunContext *Ctx, TestCond Cond, LIBC_NAMESPACE::cpp::UInt<128> LHS,
    LIBC_NAMESPACE::cpp::UInt<128> RHS, const char *LHSStr, const char *RHSStr,
    Location Loc);

template bool test<LIBC_NAMESPACE::cpp::UInt<192>>(
    RunContext *Ctx, TestCond Cond, LIBC_NAMESPACE::cpp::UInt<192> LHS,
    LIBC_NAMESPACE::cpp::UInt<192> RHS, const char *LHSStr, const char *RHSStr,
    Location Loc);

template bool test<LIBC_NAMESPACE::cpp::UInt<256>>(
    RunContext *Ctx, TestCond Cond, LIBC_NAMESPACE::cpp::UInt<256> LHS,
    LIBC_NAMESPACE::cpp::UInt<256> RHS, const char *LHSStr, const char *RHSStr,
    Location Loc);

template bool test<LIBC_NAMESPACE::cpp::UInt<320>>(
    RunContext *Ctx, TestCond Cond, LIBC_NAMESPACE::cpp::UInt<320> LHS,
    LIBC_NAMESPACE::cpp::UInt<320> RHS, const char *LHSStr, const char *RHSStr,
    Location Loc);

template bool test<LIBC_NAMESPACE::cpp::string_view>(
    RunContext *Ctx, TestCond Cond, LIBC_NAMESPACE::cpp::string_view LHS,
    LIBC_NAMESPACE::cpp::string_view RHS, const char *LHSStr,
    const char *RHSStr, Location Loc);

template bool test<LIBC_NAMESPACE::cpp::string>(RunContext *Ctx, TestCond Cond,
                                                LIBC_NAMESPACE::cpp::string LHS,
                                                LIBC_NAMESPACE::cpp::string RHS,
                                                const char *LHSStr,
                                                const char *RHSStr,
                                                Location Loc);

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
} // namespace LIBC_NAMESPACE
