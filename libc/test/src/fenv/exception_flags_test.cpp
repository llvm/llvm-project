//===-- Unittests for fegetexceptflag and fesetexceptflag -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fegetexceptflag.h"
#include "src/fenv/fesetexceptflag.h"

#include "src/__support/FPUtil/FEnvImpl.h"
#include "test/UnitTest/Test.h"

#include <fenv.h>

TEST(LlvmLibcFenvTest, GetExceptFlagAndSetExceptFlag) {
  // We will disable all exceptions to prevent invocation of the exception
  // handler.
  __llvm_libc::fputil::disable_except(FE_ALL_EXCEPT);
  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);

  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};

  for (int e : excepts) {
    // The overall idea is to raise an except and save the exception flags.
    // Next, clear the flags and then set the saved exception flags. This
    // should set the flag corresponding to the previously raised exception.
    __llvm_libc::fputil::raise_except(e);
    // Make sure that the exception flag is set.
    ASSERT_NE(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT) & e, 0);

    fexcept_t eflags;
    ASSERT_EQ(__llvm_libc::fegetexceptflag(&eflags, FE_ALL_EXCEPT), 0);

    __llvm_libc::fputil::clear_except(e);
    ASSERT_EQ(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT) & e, 0);

    ASSERT_EQ(__llvm_libc::fesetexceptflag(&eflags, FE_ALL_EXCEPT), 0);
    ASSERT_NE(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT) & e, 0);

    // Cleanup. We clear all excepts as raising excepts like FE_OVERFLOW
    // can also raise FE_INEXACT.
    __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  }

  // Next, we will raise one exception and save the flags.
  __llvm_libc::fputil::raise_except(FE_INVALID);
  fexcept_t eflags;
  __llvm_libc::fegetexceptflag(&eflags, FE_ALL_EXCEPT);
  // Clear all exceptions and raise two other exceptions.
  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  __llvm_libc::fputil::raise_except(FE_OVERFLOW | FE_INEXACT);
  // When we set the flags and test, we should only see FE_INVALID.
  __llvm_libc::fesetexceptflag(&eflags, FE_ALL_EXCEPT);
  EXPECT_EQ(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT), FE_INVALID);
}
