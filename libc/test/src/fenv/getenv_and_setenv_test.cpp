//===-- Unittests for fegetenv and fesetenv -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fegetenv.h"
#include "src/fenv/fegetround.h"
#include "src/fenv/fesetenv.h"
#include "src/fenv/fesetround.h"

#include "src/__support/FPUtil/FEnvImpl.h"
#include "utils/UnitTest/Test.h"

#include <fenv.h>

TEST(LlvmLibcFenvTest, GetEnvAndSetEnv) {
  // We will disable all exceptions to prevent invocation of the exception
  // handler.
  __llvm_libc::fputil::disable_except(FE_ALL_EXCEPT);

  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};

  for (int e : excepts) {
    __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);

    // Save the cleared environment.
    fenv_t env;
    ASSERT_EQ(__llvm_libc::fegetenv(&env), 0);

    __llvm_libc::fputil::raise_except(e);
    // Make sure that the exception is raised.
    ASSERT_NE(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT) & e, 0);

    ASSERT_EQ(__llvm_libc::fesetenv(&env), 0);
    ASSERT_EQ(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT) & e, 0);
  }
}

TEST(LlvmLibcFenvTest, Set_FE_DFL_ENV) {
  // We will disable all exceptions to prevent invocation of the exception
  // handler.
  __llvm_libc::fputil::disable_except(FE_ALL_EXCEPT);

  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};

  for (int e : excepts) {
    __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);

    // Save the cleared environment.
    fenv_t env;
    ASSERT_EQ(__llvm_libc::fegetenv(&env), 0);

    __llvm_libc::fputil::raise_except(e);
    // Make sure that the exception is raised.
    ASSERT_NE(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT) & e, 0);

    ASSERT_EQ(__llvm_libc::fesetenv(FE_DFL_ENV), 0);
    // Setting the default env should clear all exceptions.
    ASSERT_EQ(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT) & e, 0);
  }

  ASSERT_EQ(__llvm_libc::fesetround(FE_DOWNWARD), 0);
  ASSERT_EQ(__llvm_libc::fesetenv(FE_DFL_ENV), 0);
  // Setting the default env should set rounding mode to FE_TONEAREST.
  int rm = __llvm_libc::fegetround();
  EXPECT_EQ(rm, FE_TONEAREST);
}

#ifdef _WIN32
TEST(LlvmLibcFenvTest, Windows_Set_Get_Test) {
  // If a valid fenv_t is written, then reading it back out should be identical.
  fenv_t setEnv = {0x7e00053e, 0x0f00000f};
  fenv_t getEnv;
  ASSERT_EQ(__llvm_libc::fesetenv(&setEnv), 0);
  ASSERT_EQ(__llvm_libc::fegetenv(&getEnv), 0);

  ASSERT_EQ(setEnv._Fe_ctl, getEnv._Fe_ctl);
  ASSERT_EQ(setEnv._Fe_stat, getEnv._Fe_stat);
}
#endif
