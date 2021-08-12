//===-- Unittests for feholdexcept with exceptions enabled ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feholdexcept.h"

#include "src/__support/FPUtil/FEnvUtils.h"
#include "src/__support/FPUtil/FPExceptMatcher.h"
#include "utils/UnitTest/Test.h"

#include <fenv.h>

TEST(LlvmLibcFEnvTest, RaiseAndCrash) {
  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};

  for (int e : excepts) {
    fenv_t env;
    __llvm_libc::fputil::disableExcept(FE_ALL_EXCEPT);
    __llvm_libc::fputil::enableExcept(e);
    ASSERT_EQ(__llvm_libc::fputil::clearExcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(__llvm_libc::feholdexcept(&env), 0);
    // feholdexcept should disable all excepts so raising an exception
    // should not crash/invoke the exception handler.
    ASSERT_EQ(__llvm_libc::fputil::raiseExcept(e), 0);

    ASSERT_RAISES_FP_EXCEPT([=] {
      // When we put back the saved env, which has the exception enabled, it
      // should crash with SIGFPE. Note that we set the old environment
      // back inside this closure because in some test frameworks like Fuchsia's
      // zxtest, this test translates to a death test in which this closure is
      // run in a different thread. So, we set the old environment inside
      // this closure so that the exception gets enabled for the thread running
      // this closure.
      __llvm_libc::fputil::setEnv(&env);
      __llvm_libc::fputil::raiseExcept(e);
    });

    // Cleanup
    __llvm_libc::fputil::disableExcept(FE_ALL_EXCEPT);
    ASSERT_EQ(__llvm_libc::fputil::clearExcept(FE_ALL_EXCEPT), 0);
  }
}
