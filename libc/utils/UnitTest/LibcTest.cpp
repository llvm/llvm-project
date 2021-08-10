//===-- Implementation of the base class for libc unittests----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcTest.h"

#include "utils/testutils/ExecuteFunction.h"
#include <cassert>
#include <iostream>
#include <string>

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

// When the value is of integral type, just display it as normal.
template <typename ValType>
cpp::EnableIfType<cpp::IsIntegral<ValType>::Value, std::string>
describeValue(ValType Value) {
  return std::to_string(Value);
}

std::string describeValue(std::string Value) { return std::string(Value); }

// When the value is __uint128_t, also show its hexadecimal digits.
// Using template to force exact match, prevent ambiguous promotion.
template <> std::string describeValue<__uint128_t>(__uint128_t Value) {
  std::string S(sizeof(__uint128_t) * 2, '0');

  for (auto I = S.rbegin(), End = S.rend(); I != End; ++I, Value >>= 4) {
    unsigned char Mod = static_cast<unsigned char>(Value) & 15;
    *I = Mod < 10 ? '0' + Mod : 'a' + Mod - 10;
  }

  return "0x" + S;
}

template <typename ValType>
void explainDifference(ValType LHS, ValType RHS, const char *LHSStr,
                       const char *RHSStr, const char *File, unsigned long Line,
                       std::string OpString) {
  size_t OffsetLength = OpString.size() > 2 ? OpString.size() - 2 : 0;
  std::string Offset(OffsetLength, ' ');

  std::cout << File << ":" << Line << ": FAILURE\n"
            << Offset << "Expected: " << LHSStr << '\n'
            << Offset << "Which is: " << describeValue(LHS) << '\n'
            << "To be " << OpString << ": " << RHSStr << '\n'
            << Offset << "Which is: " << describeValue(RHS) << '\n';
}

template <typename ValType>
bool test(RunContext *Ctx, TestCondition Cond, ValType LHS, ValType RHS,
          const char *LHSStr, const char *RHSStr, const char *File,
          unsigned long Line) {
  auto ExplainDifference = [=](std::string OpString) {
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
    std::cout << "Unexpected test condition.\n";
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
    std::string StrTestName(TestName);
    constexpr auto GREEN = "\033[32m";
    constexpr auto RED = "\033[31m";
    constexpr auto RESET = "\033[0m";
    if ((TestFilter != nullptr) && (StrTestName != TestFilter)) {
      continue;
    }
    std::cout << GREEN << "[ RUN      ] " << RESET << TestName << '\n';
    RunContext Ctx;
    T->SetUp();
    T->setContext(&Ctx);
    T->Run();
    T->TearDown();
    auto Result = Ctx.status();
    switch (Result) {
    case RunContext::Result_Fail:
      std::cout << RED << "[  FAILED  ] " << RESET << TestName << '\n';
      ++FailCount;
      break;
    case RunContext::Result_Pass:
      std::cout << GREEN << "[       OK ] " << RESET << TestName << '\n';
      break;
    }
    ++TestCount;
  }

  if (TestCount > 0) {
    std::cout << "Ran " << TestCount << " tests. "
              << " PASS: " << TestCount - FailCount << ' '
              << " FAIL: " << FailCount << '\n';
  } else {
    std::cout << "No tests run.\n";
    if (TestFilter) {
      std::cout << "No matching test for " << TestFilter << '\n';
    }
  }

  return FailCount > 0 || TestCount == 0 ? 1 : 0;
}

template bool Test::test<char, 0>(TestCondition Cond, char LHS, char RHS,
                                  const char *LHSStr, const char *RHSStr,
                                  const char *File, unsigned long Line);

template bool Test::test<short, 0>(TestCondition Cond, short LHS, short RHS,
                                   const char *LHSStr, const char *RHSStr,
                                   const char *File, unsigned long Line);

template bool Test::test<int, 0>(TestCondition Cond, int LHS, int RHS,
                                 const char *LHSStr, const char *RHSStr,
                                 const char *File, unsigned long Line);

template bool Test::test<long, 0>(TestCondition Cond, long LHS, long RHS,
                                  const char *LHSStr, const char *RHSStr,
                                  const char *File, unsigned long Line);

template bool Test::test<long long, 0>(TestCondition Cond, long long LHS,
                                       long long RHS, const char *LHSStr,
                                       const char *RHSStr, const char *File,
                                       unsigned long Line);

template bool Test::test<unsigned char, 0>(TestCondition Cond,
                                           unsigned char LHS, unsigned char RHS,
                                           const char *LHSStr,
                                           const char *RHSStr, const char *File,
                                           unsigned long Line);

template bool
Test::test<unsigned short, 0>(TestCondition Cond, unsigned short LHS,
                              unsigned short RHS, const char *LHSStr,
                              const char *RHSStr, const char *File,
                              unsigned long Line);

template bool Test::test<unsigned int, 0>(TestCondition Cond, unsigned int LHS,
                                          unsigned int RHS, const char *LHSStr,
                                          const char *RHSStr, const char *File,
                                          unsigned long Line);

template bool Test::test<unsigned long, 0>(TestCondition Cond,
                                           unsigned long LHS, unsigned long RHS,
                                           const char *LHSStr,
                                           const char *RHSStr, const char *File,
                                           unsigned long Line);

template bool Test::test<bool, 0>(TestCondition Cond, bool LHS, bool RHS,
                                  const char *LHSStr, const char *RHSStr,
                                  const char *File, unsigned long Line);

template bool
Test::test<unsigned long long, 0>(TestCondition Cond, unsigned long long LHS,
                                  unsigned long long RHS, const char *LHSStr,
                                  const char *RHSStr, const char *File,
                                  unsigned long Line);

template bool Test::test<__uint128_t, 0>(TestCondition Cond, __uint128_t LHS,
                                         __uint128_t RHS, const char *LHSStr,
                                         const char *RHSStr, const char *File,
                                         unsigned long Line);

bool Test::testStrEq(const char *LHS, const char *RHS, const char *LHSStr,
                     const char *RHSStr, const char *File, unsigned long Line) {
  return internal::test(Ctx, Cond_EQ, LHS ? std::string(LHS) : std::string(),
                        RHS ? std::string(RHS) : std::string(), LHSStr, RHSStr,
                        File, Line);
}

bool Test::testStrNe(const char *LHS, const char *RHS, const char *LHSStr,
                     const char *RHSStr, const char *File, unsigned long Line) {
  return internal::test(Ctx, Cond_NE, LHS ? std::string(LHS) : std::string(),
                        RHS ? std::string(RHS) : std::string(), LHSStr, RHSStr,
                        File, Line);
}

bool Test::testMatch(bool MatchResult, MatcherBase &Matcher, const char *LHSStr,
                     const char *RHSStr, const char *File, unsigned long Line) {
  if (MatchResult)
    return true;

  Ctx->markFail();
  std::cout << File << ":" << Line << ": FAILURE\n"
            << "Failed to match " << LHSStr << " against " << RHSStr << ".\n";
  testutils::StreamWrapper OutsWrapper = testutils::outs();
  Matcher.explainError(OutsWrapper);
  return false;
}

#ifdef ENABLE_SUBPROCESS_TESTS

bool Test::testProcessKilled(testutils::FunctionCaller *Func, int Signal,
                             const char *LHSStr, const char *RHSStr,
                             const char *File, unsigned long Line) {
  testutils::ProcessStatus Result = testutils::invokeInSubprocess(Func, 500);

  if (const char *error = Result.getError()) {
    Ctx->markFail();
    std::cout << File << ":" << Line << ": FAILURE\n" << error << '\n';
    return false;
  }

  if (Result.timedOut()) {
    Ctx->markFail();
    std::cout << File << ":" << Line << ": FAILURE\n"
              << "Process timed out after " << 500 << " milliseconds.\n";
    return false;
  }

  if (Result.exitedNormally()) {
    Ctx->markFail();
    std::cout << File << ":" << Line << ": FAILURE\n"
              << "Expected " << LHSStr
              << " to be killed by a signal\nBut it exited normally!\n";
    return false;
  }

  int KilledBy = Result.getFatalSignal();
  assert(KilledBy != 0 && "Not killed by any signal");
  if (Signal == -1 || KilledBy == Signal)
    return true;

  using testutils::signalAsString;
  Ctx->markFail();
  std::cout << File << ":" << Line << ": FAILURE\n"
            << "              Expected: " << LHSStr << '\n'
            << "To be killed by signal: " << Signal << '\n'
            << "              Which is: " << signalAsString(Signal) << '\n'
            << "  But it was killed by: " << KilledBy << '\n'
            << "              Which is: " << signalAsString(KilledBy) << '\n';
  return false;
}

bool Test::testProcessExits(testutils::FunctionCaller *Func, int ExitCode,
                            const char *LHSStr, const char *RHSStr,
                            const char *File, unsigned long Line) {
  testutils::ProcessStatus Result = testutils::invokeInSubprocess(Func, 500);

  if (const char *error = Result.getError()) {
    Ctx->markFail();
    std::cout << File << ":" << Line << ": FAILURE\n" << error << '\n';
    return false;
  }

  if (Result.timedOut()) {
    Ctx->markFail();
    std::cout << File << ":" << Line << ": FAILURE\n"
              << "Process timed out after " << 500 << " milliseconds.\n";
    return false;
  }

  if (!Result.exitedNormally()) {
    Ctx->markFail();
    std::cout << File << ":" << Line << ": FAILURE\n"
              << "Expected " << LHSStr << '\n'
              << "to exit with exit code " << ExitCode << '\n'
              << "But it exited abnormally!\n";
    return false;
  }

  int ActualExit = Result.getExitCode();
  if (ActualExit == ExitCode)
    return true;

  Ctx->markFail();
  std::cout << File << ":" << Line << ": FAILURE\n"
            << "Expected exit code of: " << LHSStr << '\n'
            << "             Which is: " << ActualExit << '\n'
            << "       To be equal to: " << RHSStr << '\n'
            << "             Which is: " << ExitCode << '\n';
  return false;
}

#endif // ENABLE_SUBPROCESS_TESTS
} // namespace testing
} // namespace __llvm_libc
