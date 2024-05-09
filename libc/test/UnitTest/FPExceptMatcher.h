//===-- FPExceptMatcher.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_FPEXCEPTMATCHER_H
#define LLVM_LIBC_TEST_UNITTEST_FPEXCEPTMATCHER_H

#include "hdr/fenv_macros.h"
#include "test/UnitTest/Test.h"
#include "test/UnitTest/TestLogger.h"

#if LIBC_TEST_HAS_MATCHERS()

namespace LIBC_NAMESPACE {
namespace testing {

// Used to compare FP exception flag states with nice error printing
class FPExceptMatcher : public Matcher<int> {
  const int expected;
  int actual;

public:
  explicit FPExceptMatcher(int expected) : expected(expected) {}

  void explainError() override {
    tlog << "Expected floating point exceptions: " << expected << ' ';
    printExcepts(expected);
    tlog << '\n';

    tlog << "Actual floating point exceptions: " << actual << ' ';
    printExcepts(actual);
    tlog << '\n';
  }

  bool match(int got) {
    actual = got;
    return got == expected;
  }

private:
  void printExcepts(int excepts) {
    if (!excepts) {
      tlog << "(no exceptions)";
      return;
    }

    bool firstPrinted = false;
    auto printWithPipe = [&](const char *name) {
      if (firstPrinted)
        tlog << " | ";

      tlog << name;

      firstPrinted = true;
    };

    tlog << '(';

    if (FE_DIVBYZERO & excepts)
      printWithPipe("FE_DIVBYZERO");

    if (FE_INEXACT & excepts)
      printWithPipe("FE_INEXACT");

    if (FE_INVALID & excepts)
      printWithPipe("FE_INVALID");

    if (FE_OVERFLOW & excepts)
      printWithPipe("FE_OVERFLOW");

    if (FE_UNDERFLOW & excepts)
      printWithPipe("FE_UNDERFLOW");

    tlog << ')';
  }
};

// TODO: Make the matcher match specific exceptions instead of just identifying
// that an exception was raised.
// Used in death tests for fenv
class FPExceptCallableMatcher : public Matcher<bool> {
  bool exceptionRaised;

public:
  class FunctionCaller {
  public:
    virtual ~FunctionCaller() {}
    virtual void call() = 0;
  };

  template <typename Func> static FunctionCaller *getFunctionCaller(Func func) {
    struct Callable : public FunctionCaller {
      Func f;
      explicit Callable(Func theFunc) : f(theFunc) {}
      void call() override { f(); }
    };

    return new Callable(func);
  }

  // Takes ownership of func.
  explicit FPExceptCallableMatcher(FunctionCaller *func);

  bool match(bool unused) { return exceptionRaised; }

  void explainError() override {
    tlog << "A floating point exception should have been raised but it "
         << "wasn't\n";
  }
};

} // namespace testing
} // namespace LIBC_NAMESPACE

// Matches on the FP exception flag `expected` being *equal* to FP exception
// flag `actual`
#define EXPECT_FP_EXCEPT_EQUAL(expected, actual)                               \
  EXPECT_THAT((actual), LIBC_NAMESPACE::testing::FPExceptMatcher((expected)))

#define ASSERT_FP_EXCEPT_EQUAL(expected, actual)                               \
  ASSERT_THAT((actual), LIBC_NAMESPACE::testing::FPExceptMatcher((expected)))

#define ASSERT_RAISES_FP_EXCEPT(func)                                          \
  ASSERT_THAT(                                                                 \
      true,                                                                    \
      LIBC_NAMESPACE::testing::FPExceptCallableMatcher(                        \
          LIBC_NAMESPACE::testing::FPExceptCallableMatcher::getFunctionCaller( \
              func)))

// Does not return the value of `expr_or_statement`, i.e., intended usage
// is: `EXPECT_FP_EXCEPT(FE_INVALID, EXPECT_FP_EQ(..., ...));` or
// ```
// EXPECT_FP_EXCEPT(FE_ALL_EXCEPT, {
//   stmt;
//   ...
// });
// ```
// Ensures that fp excepts are cleared before executing `expr_or_statement`
// Checking (expected = 0) should ensure that no exceptions were set
#define EXPECT_FP_EXCEPT(expected, expr_or_statement)                          \
  do {                                                                         \
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);                       \
    expr_or_statement;                                                         \
    int expected_ = (expected);                                                \
    int mask_ = expected_ ? expected_ : FE_ALL_EXCEPT;                         \
    if (math_errhandling & MATH_ERREXCEPT) {                                   \
      EXPECT_FP_EXCEPT_EQUAL(expected_,                                        \
                             LIBC_NAMESPACE::fputil::test_except(mask_));      \
    }                                                                          \
  } while (0)

#else // !LIBC_TEST_HAS_MATCHERS()

#define ASSERT_RAISES_FP_EXCEPT(func) ASSERT_DEATH(func, WITH_SIGNAL(SIGFPE))

#endif // LIBC_TEST_HAS_MATCHERS()

#endif // LLVM_LIBC_TEST_UNITTEST_FPEXCEPTMATCHER_H
