//===-- Unittests for fe{get|set|test}exceptflag --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/fexcept_t.h"
#include "src/fenv/fegetexceptflag.h"
#include "src/fenv/fesetexceptflag.h"
#include "src/fenv/fetestexceptflag.h"

#include "src/__support/FPUtil/FEnvImpl.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/Test.h"

#include "excepts.h"

using LlvmLibcFEnvTest = LIBC_NAMESPACE::testing::FEnvSafeTest;

TEST_F(LlvmLibcFEnvTest, GetSetTestExceptFlag) {
  // We will disable all exceptions to prevent invocation of the exception
  // handler.
  LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);

  for (int e : EXCEPTS) {
    // The overall idea is to raise an except and save the exception flags.
    // Next, clear the flags and then set the saved exception flags. This
    // should set the flag corresponding to the previously raised exception.
    LIBC_NAMESPACE::fputil::raise_except(e);
    // Make sure that the exception flag is set.
    ASSERT_NE(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) & e, 0);

    fexcept_t eflags;
    ASSERT_EQ(LIBC_NAMESPACE::fegetexceptflag(&eflags, FE_ALL_EXCEPT), 0);

    LIBC_NAMESPACE::fputil::clear_except(e);
    ASSERT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) & e, 0);

    ASSERT_EQ(LIBC_NAMESPACE::fesetexceptflag(&eflags, FE_ALL_EXCEPT), 0);
    ASSERT_NE(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) & e, 0);

    // Exception flags are exactly the flags corresponding to the previously
    // raised exception.
    ASSERT_EQ(LIBC_NAMESPACE::fetestexceptflag(&eflags, FE_ALL_EXCEPT),
              LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT));

    // Cleanup. We clear all excepts as raising excepts like FE_OVERFLOW
    // can also raise FE_INEXACT.
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  }

  // Next, we will raise one exception, save the flag and clear all exceptions.
  LIBC_NAMESPACE::fputil::raise_except(FE_INVALID);
  fexcept_t invalid_flag;
  LIBC_NAMESPACE::fegetexceptflag(&invalid_flag, FE_ALL_EXCEPT);
  ASSERT_EQ(LIBC_NAMESPACE::fetestexceptflag(&invalid_flag, FE_ALL_EXCEPT),
            FE_INVALID);
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);

  // Raise two other exceptions and verify that they are set.
  LIBC_NAMESPACE::fputil::raise_except(FE_OVERFLOW | FE_INEXACT);
  fexcept_t overflow_and_inexact_flag;
  LIBC_NAMESPACE::fegetexceptflag(&overflow_and_inexact_flag, FE_ALL_EXCEPT);
  ASSERT_EQ(LIBC_NAMESPACE::fetestexceptflag(&overflow_and_inexact_flag,
                                             FE_ALL_EXCEPT),
            FE_OVERFLOW | FE_INEXACT);
  ASSERT_EQ(LIBC_NAMESPACE::fetestexceptflag(&overflow_and_inexact_flag,
                                             FE_OVERFLOW | FE_INEXACT),
            FE_OVERFLOW | FE_INEXACT);

  // When we set the flags and test, we should only see FE_INVALID.
  LIBC_NAMESPACE::fesetexceptflag(&invalid_flag, FE_ALL_EXCEPT);
  EXPECT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT), FE_INVALID);
}
