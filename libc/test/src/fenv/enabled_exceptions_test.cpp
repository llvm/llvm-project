//===-- Unittests for feraiseexcept with exceptions enabled ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feclearexcept.h"
#include "src/fenv/feraiseexcept.h"
#include "src/fenv/fetestexcept.h"

#include "utils/FPUtil/FEnv.h"
#include "utils/UnitTest/Test.h"

#include <fenv.h>
#include <signal.h>

// This test enables an exception and verifies that raising that exception
// triggers SIGFPE.
TEST(LlvmLibcExceptionStatusTest, RaiseAndCrash) {
  // TODO: Install a floating point exception handler and verify that the
  // the expected exception was raised. One will have to longjmp back from
  // that exception handler, so such a testing can be done after we have
  // longjmp implemented.

  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};

  // We '|' the individual exception flags instead of using FE_ALL_EXCEPT
  // as it can include non-standard extensions. Note that we should be able
  // to compile this file with headers from other libcs as well.
  constexpr int allExcepts =
      FE_DIVBYZERO | FE_INVALID | FE_INEXACT | FE_OVERFLOW | FE_UNDERFLOW;

  for (int e : excepts) {
    ASSERT_DEATH(
        [=] {
          __llvm_libc::fputil::disableExcept(FE_ALL_EXCEPT);
          __llvm_libc::fputil::enableExcept(e);
          ASSERT_EQ(__llvm_libc::feclearexcept(FE_ALL_EXCEPT), 0);
          // Raising all exceptions except |e| should not call the
          // SIGFPE handler. They should set the exception flag though,
          // so we verify that.
          int others = allExcepts & ~e;
          ASSERT_EQ(__llvm_libc::feraiseexcept(others), 0);
          ASSERT_EQ(__llvm_libc::fetestexcept(others), others);

          // We don't check the return value when raising |e| as
          // feraiseexcept will not return when it raises an enabled
          // exception.
          __llvm_libc::feraiseexcept(e);
        },
        WITH_SIGNAL(SIGFPE));
  }
}
