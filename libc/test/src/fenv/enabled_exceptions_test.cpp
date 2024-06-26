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

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/macros/properties/architectures.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPExceptMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/fenv_macros.h"
#include <signal.h>

#include "excepts.h"

using LlvmLibcExceptionStatusTest = LIBC_NAMESPACE::testing::FEnvSafeTest;

// This test enables an exception and verifies that raising that exception
// triggers SIGFPE.
TEST_F(LlvmLibcExceptionStatusTest, RaiseAndCrash) {
#if defined(LIBC_TARGET_ARCH_IS_ANY_ARM) ||                                    \
    defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
  // Few Arm HW implementations do not trap exceptions. We skip this test
  // completely on such HW.
  //
  // Whether HW supports trapping exceptions or not is deduced by enabling an
  // exception and reading back to see if the exception got enabled. If the
  // exception did not get enabled, then it means that the HW does not support
  // trapping exceptions.
  LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);
  LIBC_NAMESPACE::fputil::enable_except(FE_DIVBYZERO);
  if (LIBC_NAMESPACE::fputil::get_except() == 0)
    return;
#endif // Architectures where exception trapping is not supported

  // TODO: Install a floating point exception handler and verify that the
  // the expected exception was raised. One will have to longjmp back from
  // that exception handler, so such a testing can be done after we have
  // longjmp implemented.

  for (int e : EXCEPTS) {
    LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);
    LIBC_NAMESPACE::fputil::enable_except(e);
    ASSERT_EQ(LIBC_NAMESPACE::feclearexcept(FE_ALL_EXCEPT), 0);
    // Raising all exceptions except |e| should not call the
    // SIGFPE handler. They should set the exception flag though,
    // so we verify that. Since other exceptions like FE_DIVBYZERO
    // can raise FE_INEXACT as well, we don't verify the other
    // exception flags when FE_INEXACT is enabled.
    if (e != FE_INEXACT) {
      int others = ALL_EXCEPTS & ~e;
      ASSERT_EQ(LIBC_NAMESPACE::feraiseexcept(others), 0);
      ASSERT_EQ(LIBC_NAMESPACE::fetestexcept(others), others);
    }

    ASSERT_RAISES_FP_EXCEPT([=] {
      // In test frameworks like Fuchsia's zxtest, this translates to
      // a death test which runs this closure in a different thread. So,
      // we enable the exception again inside this closure so that the
      // exception gets enabled for the thread running this closure.
      LIBC_NAMESPACE::fputil::enable_except(e);
      LIBC_NAMESPACE::feraiseexcept(e);
    });

    // Cleanup.
    LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);
    ASSERT_EQ(LIBC_NAMESPACE::feclearexcept(FE_ALL_EXCEPT), 0);
  }
}
