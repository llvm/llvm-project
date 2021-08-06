//===-- Unittests for fegetenv and fesetenv -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fegetenv.h"
#include "src/fenv/fesetenv.h"

#include "src/__support/FPUtil/FEnvUtils.h"
#include "utils/UnitTest/Test.h"

#include <fenv.h>

TEST(LlvmLibcFenvTest, GetEnvAndSetEnv) {
  // We will disable all exceptions to prevent invocation of the exception
  // handler.
  __llvm_libc::fputil::disableExcept(FE_ALL_EXCEPT);

  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};

  for (int e : excepts) {
    __llvm_libc::fputil::clearExcept(FE_ALL_EXCEPT);

    // Save the cleared environment.
    fenv_t env;
    ASSERT_EQ(__llvm_libc::fegetenv(&env), 0);

    __llvm_libc::fputil::raiseExcept(e);
    // Make sure that the exception is raised.
    ASSERT_NE(__llvm_libc::fputil::testExcept(FE_ALL_EXCEPT) & e, 0);

    ASSERT_EQ(__llvm_libc::fesetenv(&env), 0);
    ASSERT_EQ(__llvm_libc::fputil::testExcept(FE_ALL_EXCEPT) & e, 0);
  }
}
