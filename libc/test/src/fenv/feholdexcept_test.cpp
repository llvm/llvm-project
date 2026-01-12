//===-- Unittests for feholdexcept with exceptions enabled ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#undef LIBC_MATH_USE_SYSTEM_FENV

#include "hdr/types/fenv_t.h"
#include "src/fenv/feholdexcept.h"

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/macros/properties/architectures.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPExceptMatcher.h"
#include "test/UnitTest/Test.h"

#include "excepts.h"

using LlvmLibcFEnvTest = LIBC_NAMESPACE::testing::FEnvSafeTest;

TEST_F(LlvmLibcFEnvTest, RaiseAndCrash) {
  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};

  for (int e : excepts) {
    fenv_t env;
    LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);
    LIBC_NAMESPACE::fputil::enable_except(e);
    ASSERT_EQ(LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(LIBC_NAMESPACE::feholdexcept(&env), 0);
    // feholdexcept should disable all excepts so raising an exception
    // should not crash/invoke the exception handler.
    ASSERT_EQ(LIBC_NAMESPACE::fputil::raise_except(e), 0);

#if defined(LIBC_TRAP_ON_RAISE_FP_EXCEPT) && defined(__SSE__)
    ASSERT_RAISES_FP_EXCEPT([=] {
      // When we put back the saved env, which has the exception enabled, it
      // should crash with SIGFPE. Note that we set the old environment
      // back inside this closure because in some test frameworks like Fuchsia's
      // zxtest, this test translates to a death test in which this closure is
      // run in a different thread. So, we set the old environment inside
      // this closure so that the exception gets enabled for the thread running
      // this closure.
      LIBC_NAMESPACE::fputil::set_env(&env);
      LIBC_NAMESPACE::fputil::raise_except(e);
    });
#endif // LIBC_TRAP_ON_RAISE_FP_EXCEPT

    // Cleanup
    LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);
    ASSERT_EQ(LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT), 0);
  }
}
