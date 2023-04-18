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
#include "test/UnitTest/ExecuteFunction.h"
#include "test/UnitTest/TestLogger.h"
#include <cassert>

namespace __llvm_libc {
namespace testing {

// This need not be a class as all it has is a single read-write state variable.
// But, we make it class as then its implementation can be hidden from the
// header file.
class RunContext {
public:
  enum RunResult { Result_Pass = 1, Result_Fail = 2 };

  RunResult status() const { return Status; }

  void markFail() { Status = Result_Fail; }

private:
  RunResult Status = Result_Pass;
};

namespace internal {

// When the value is UInt128 or __uint128_t, show its hexadecimal digits.
// We cannot just use a UInt128 specialization as that resolves to only
// one type, UInt<128> or __uint128_t. We want both overloads as we want to
// be able to unittest UInt<128> on platforms where UInt128 resolves to
// UInt128.
template <typename T>
cpp::enable_if_t<cpp::is_integral_v<T> && cpp::is_unsigned_v<T>, cpp::string>
describeValueUInt(T Value) {
  static_assert(sizeof(T) % 8 == 0, "Unsupported size of UInt");
  cpp::string S(sizeof(T) * 2, '0');

  constexpr char HEXADECIMALS[16] = {'0', '1', '2', '3', '4', '5', '6', '7',
                                     '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  const size_t Size = S.size();
  for (size_t I = 0; I < Size; I += 2, Value >>= 8) {
    unsigned char Mod = static_cast<unsigned char>(Value) & 0xFF;
    S[Size - I] = HEXADECIMALS[Mod & 0x0F];
    S[Size - (I + 1)] = HEXADECIMALS[Mod & 0x0F];
  }

  return "0x" + S;
}

// When the value is of integral type, just display it as normal.
template <typename ValType>
cpp::enable_if_t<cpp::is_integral_v<ValType>, cpp::string>
describeValue(ValType Value) {
  if constexpr (sizeof(ValType) <= sizeof(uint64_t)) {
    return cpp::to_string(Value);
  } else {
    return describeValueUInt(Value);
  }
}

cpp::string describeValue(cpp::string Value) { return Value; }
cpp::string_view describeValue(cpp::string_view Value) { return Value; }

template <typename ValType>
void explainDifference(ValType LHS, ValType RHS, const char *LHSStr,
                       const char *RHSStr, const char *File, unsigned long Line,
                       cpp::string OpString) {
  size_t OffsetLength = OpString.size() > 2 ? OpString.size() - 2 : 0;
  cpp::string Offset(OffsetLength, ' ');

  tlog << File << ":" << Line << ": FAILURE\n"
       << Offset << "Expected: " << LHSStr << '\n'
       << Offset << "Which is: " << describeValue(LHS) << '\n'
       << "To be " << OpString << ": " << RHSStr << '\n'
       << Offset << "Which is: " << describeValue(RHS) << '\n';
}

template <typename ValType>
bool test(RunContext *Ctx, TestCondition Cond, ValType LHS, ValType RHS,
          const char *LHSStr, const char *RHSStr, const char *File,
          unsigned long Line) {
  auto ExplainDifference = [=](cpp::string OpString) {
    explainDifference(LHS, RHS, LHSStr, RHSStr, File, Line, OpString);
  };

  switch (Cond) {
  case Cond_EQ:
    if (LHS == RHS)
      return true;

    Ctx->markFail();
    ExplainDifference("equal to");
    return false;
  case Cond_NE:
    if (LHS != RHS)
      return true;

    Ctx->markFail();
    ExplainDifference("not equal to");
    return false;
  case Cond_LT:
    if (LHS < RHS)
      return true;

    Ctx->markFail();
    ExplainDifference("less than");
    return false;
  case Cond_LE:
    if (LHS <= RHS)
      return true;

    Ctx->markFail();
    ExplainDifference("less than or equal to");
    return false;
  case Cond_GT:
    if (LHS > RHS)
      return true;

    Ctx->markFail();
    ExplainDifference("greater than");
    return false;
  case Cond_GE:
    if (LHS >= RHS)
      return true;

    Ctx->markFail();
    ExplainDifference("greater than or equal to");
    return false;
  default:
    Ctx->markFail();
    tlog << "Unexpected test condition.\n";
    return false;
  }
}

} // namespace internal

Test *Test::Start = nullptr;
Test *Test::End = nullptr;

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
    RunContext Ctx;
    T->SetUp();
    T->setContext(&Ctx);
    T->Run();
    T->TearDown();
    auto Result = Ctx.status();
    switch (Result) {
    case RunContext::Result_Fail:
      tlog << RED << "[  FAILED  ] " << RESET << TestName << '\n';
      ++FailCount;
      break;
    case RunContext::Result_Pass:
      tlog << GREEN << "[       OK ] " << RESET << TestName << '\n';
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

template bool test<char>(RunContext *Ctx, TestCondition Cond, char LHS,
                         char RHS, const char *LHSStr, const char *RHSStr,
                         const char *File, unsigned long Line);

template bool test<short>(RunContext *Ctx, TestCondition Cond, short LHS,
                          short RHS, const char *LHSStr, const char *RHSStr,
                          const char *File, unsigned long Line);

template bool test<int>(RunContext *Ctx, TestCondition Cond, int LHS, int RHS,
                        const char *LHSStr, const char *RHSStr,
                        const char *File, unsigned long Line);

template bool test<long>(RunContext *Ctx, TestCondition Cond, long LHS,
                         long RHS, const char *LHSStr, const char *RHSStr,
                         const char *File, unsigned long Line);

template bool test<long long>(RunContext *Ctx, TestCondition Cond,
                              long long LHS, long long RHS, const char *LHSStr,
                              const char *RHSStr, const char *File,
                              unsigned long Line);

template bool test<unsigned char>(RunContext *Ctx, TestCondition Cond,
                                  unsigned char LHS, unsigned char RHS,
                                  const char *LHSStr, const char *RHSStr,
                                  const char *File, unsigned long Line);

template bool test<unsigned short>(RunContext *Ctx, TestCondition Cond,
                                   unsigned short LHS, unsigned short RHS,
                                   const char *LHSStr, const char *RHSStr,
                                   const char *File, unsigned long Line);

template bool test<unsigned int>(RunContext *Ctx, TestCondition Cond,
                                 unsigned int LHS, unsigned int RHS,
                                 const char *LHSStr, const char *RHSStr,
                                 const char *File, unsigned long Line);

template bool test<unsigned long>(RunContext *Ctx, TestCondition Cond,
                                  unsigned long LHS, unsigned long RHS,
                                  const char *LHSStr, const char *RHSStr,
                                  const char *File, unsigned long Line);

template bool test<bool>(RunContext *Ctx, TestCondition Cond, bool LHS,
                         bool RHS, const char *LHSStr, const char *RHSStr,
                         const char *File, unsigned long Line);

template bool test<unsigned long long>(RunContext *Ctx, TestCondition Cond,
                                       unsigned long long LHS,
                                       unsigned long long RHS,
                                       const char *LHSStr, const char *RHSStr,
                                       const char *File, unsigned long Line);

// We cannot just use a single UInt128 specialization as that resolves to only
// one type, UInt<128> or __uint128_t. We want both overloads as we want to
// be able to unittest UInt<128> on platforms where UInt128 resolves to
// UInt128.
#ifdef __SIZEOF_INT128__
// When builtin __uint128_t type is available, include its specialization
// also.
template bool test<__uint128_t>(RunContext *Ctx, TestCondition Cond,
                                __uint128_t LHS, __uint128_t RHS,
                                const char *LHSStr, const char *RHSStr,
                                const char *File, unsigned long Line);
#endif

template bool test<__llvm_libc::cpp::UInt<128>>(
    RunContext *Ctx, TestCondition Cond, __llvm_libc::cpp::UInt<128> LHS,
    __llvm_libc::cpp::UInt<128> RHS, const char *LHSStr, const char *RHSStr,
    const char *File, unsigned long Line);

template bool test<__llvm_libc::cpp::UInt<192>>(
    RunContext *Ctx, TestCondition Cond, __llvm_libc::cpp::UInt<192> LHS,
    __llvm_libc::cpp::UInt<192> RHS, const char *LHSStr, const char *RHSStr,
    const char *File, unsigned long Line);

template bool test<__llvm_libc::cpp::UInt<256>>(
    RunContext *Ctx, TestCondition Cond, __llvm_libc::cpp::UInt<256> LHS,
    __llvm_libc::cpp::UInt<256> RHS, const char *LHSStr, const char *RHSStr,
    const char *File, unsigned long Line);

template bool test<__llvm_libc::cpp::UInt<320>>(
    RunContext *Ctx, TestCondition Cond, __llvm_libc::cpp::UInt<320> LHS,
    __llvm_libc::cpp::UInt<320> RHS, const char *LHSStr, const char *RHSStr,
    const char *File, unsigned long Line);

template bool test<__llvm_libc::cpp::string_view>(
    RunContext *Ctx, TestCondition Cond, __llvm_libc::cpp::string_view LHS,
    __llvm_libc::cpp::string_view RHS, const char *LHSStr, const char *RHSStr,
    const char *File, unsigned long Line);

} // namespace internal

bool Test::testStrEq(const char *LHS, const char *RHS, const char *LHSStr,
                     const char *RHSStr, const char *File, unsigned long Line) {
  return internal::test(Ctx, Cond_EQ, LHS ? cpp::string(LHS) : cpp::string(),
                        RHS ? cpp::string(RHS) : cpp::string(), LHSStr, RHSStr,
                        File, Line);
}

bool Test::testStrNe(const char *LHS, const char *RHS, const char *LHSStr,
                     const char *RHSStr, const char *File, unsigned long Line) {
  return internal::test(Ctx, Cond_NE, LHS ? cpp::string(LHS) : cpp::string(),
                        RHS ? cpp::string(RHS) : cpp::string(), LHSStr, RHSStr,
                        File, Line);
}

bool Test::testMatch(bool MatchResult, MatcherBase &Matcher, const char *LHSStr,
                     const char *RHSStr, const char *File, unsigned long Line) {
  if (MatchResult)
    return true;

  Ctx->markFail();
  if (!Matcher.is_silent()) {
    tlog << File << ":" << Line << ": FAILURE\n"
         << "Failed to match " << LHSStr << " against " << RHSStr << ".\n";
    Matcher.explainError();
  }
  return false;
}

#ifdef ENABLE_SUBPROCESS_TESTS

bool Test::testProcessKilled(testutils::FunctionCaller *Func, int Signal,
                             const char *LHSStr, const char *RHSStr,
                             const char *File, unsigned long Line) {
  testutils::ProcessStatus Result = testutils::invoke_in_subprocess(Func, 500);

  if (const char *error = Result.get_error()) {
    Ctx->markFail();
    tlog << File << ":" << Line << ": FAILURE\n" << error << '\n';
    return false;
  }

  if (Result.timed_out()) {
    Ctx->markFail();
    tlog << File << ":" << Line << ": FAILURE\n"
         << "Process timed out after " << 500 << " milliseconds.\n";
    return false;
  }

  if (Result.exited_normally()) {
    Ctx->markFail();
    tlog << File << ":" << Line << ": FAILURE\n"
         << "Expected " << LHSStr
         << " to be killed by a signal\nBut it exited normally!\n";
    return false;
  }

  int KilledBy = Result.get_fatal_signal();
  assert(KilledBy != 0 && "Not killed by any signal");
  if (Signal == -1 || KilledBy == Signal)
    return true;

  using testutils::signal_as_string;
  Ctx->markFail();
  tlog << File << ":" << Line << ": FAILURE\n"
       << "              Expected: " << LHSStr << '\n'
       << "To be killed by signal: " << Signal << '\n'
       << "              Which is: " << signal_as_string(Signal) << '\n'
       << "  But it was killed by: " << KilledBy << '\n'
       << "              Which is: " << signal_as_string(KilledBy) << '\n';
  return false;
}

bool Test::testProcessExits(testutils::FunctionCaller *Func, int ExitCode,
                            const char *LHSStr, const char *RHSStr,
                            const char *File, unsigned long Line) {
  testutils::ProcessStatus Result = testutils::invoke_in_subprocess(Func, 500);

  if (const char *error = Result.get_error()) {
    Ctx->markFail();
    tlog << File << ":" << Line << ": FAILURE\n" << error << '\n';
    return false;
  }

  if (Result.timed_out()) {
    Ctx->markFail();
    tlog << File << ":" << Line << ": FAILURE\n"
         << "Process timed out after " << 500 << " milliseconds.\n";
    return false;
  }

  if (!Result.exited_normally()) {
    Ctx->markFail();
    tlog << File << ":" << Line << ": FAILURE\n"
         << "Expected " << LHSStr << '\n'
         << "to exit with exit code " << ExitCode << '\n'
         << "But it exited abnormally!\n";
    return false;
  }

  int ActualExit = Result.get_exit_code();
  if (ActualExit == ExitCode)
    return true;

  Ctx->markFail();
  tlog << File << ":" << Line << ": FAILURE\n"
       << "Expected exit code of: " << LHSStr << '\n'
       << "             Which is: " << ActualExit << '\n'
       << "       To be equal to: " << RHSStr << '\n'
       << "             Which is: " << ExitCode << '\n';
  return false;
}

#endif // ENABLE_SUBPROCESS_TESTS
} // namespace testing
} // namespace __llvm_libc
