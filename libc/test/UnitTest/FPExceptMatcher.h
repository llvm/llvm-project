//===-- FPExceptMatcher.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_FPEXCEPTMATCHER_H
#define LLVM_LIBC_TEST_UNITTEST_FPEXCEPTMATCHER_H

#ifndef LIBC_COPT_TEST_USE_FUCHSIA

#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {
namespace testing {

// TODO: Make the matcher match specific exceptions instead of just identifying
// that an exception was raised.
class FPExceptMatcher : public Matcher<bool> {
  bool exceptionRaised;

public:
  class FunctionCaller {
  public:
    virtual ~FunctionCaller(){};
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
  explicit FPExceptMatcher(FunctionCaller *func);

  bool match(bool unused) { return exceptionRaised; }

  void explainError() override {
    tlog << "A floating point exception should have been raised but it "
         << "wasn't\n";
  }
};

} // namespace testing
} // namespace LIBC_NAMESPACE

#define ASSERT_RAISES_FP_EXCEPT(func)                                          \
  ASSERT_THAT(                                                                 \
      true,                                                                    \
      LIBC_NAMESPACE::testing::FPExceptMatcher(                                \
          LIBC_NAMESPACE::testing::FPExceptMatcher::getFunctionCaller(func)))
#else
#define ASSERT_RAISES_FP_EXCEPT(func) ASSERT_DEATH(func, WITH_SIGNAL(SIGFPE))
#endif // LIBC_COPT_TEST_USE_FUCHSIA

#endif // LLVM_LIBC_TEST_UNITTEST_FPEXCEPTMATCHER_H
