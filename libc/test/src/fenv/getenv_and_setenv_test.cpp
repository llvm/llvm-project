//===-- Unittests for fegetenv and fesetenv -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/fenv_t.h"
#include "src/fenv/fegetenv.h"
#include "src/fenv/fegetround.h"
#include "src/fenv/fesetenv.h"
#include "src/fenv/fesetround.h"

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/macros/properties/os.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/Test.h"

#include "excepts.h"

using LlvmLibcFEnvTest = LIBC_NAMESPACE::testing::FEnvSafeTest;

#ifndef LIBC_TARGET_OS_IS_WINDOWS
TEST_F(LlvmLibcFEnvTest, GetEnvAndSetEnv) {
  // We will disable all exceptions to prevent invocation of the exception
  // handler.
  LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);

  for (int e : EXCEPTS) {
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);

    // Save the cleared environment.
    fenv_t env;
    ASSERT_EQ(LIBC_NAMESPACE::fegetenv(&env), 0);

    LIBC_NAMESPACE::fputil::raise_except(e);
    // Make sure that the exception is raised.
    ASSERT_NE(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) & e, 0);

    ASSERT_EQ(LIBC_NAMESPACE::fesetenv(&env), 0);
    ASSERT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) & e, 0);
  }
}

TEST_F(LlvmLibcFEnvTest, Set_FE_DFL_ENV) {
  // We will disable all exceptions to prevent invocation of the exception
  // handler.
  LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);

  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};

  for (int e : excepts) {
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);

    // Save the cleared environment.
    fenv_t env;
    ASSERT_EQ(LIBC_NAMESPACE::fegetenv(&env), 0);

    LIBC_NAMESPACE::fputil::raise_except(e);
    // Make sure that the exception is raised.
    ASSERT_NE(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) & e, 0);

    ASSERT_EQ(LIBC_NAMESPACE::fesetenv(FE_DFL_ENV), 0);
    // Setting the default env should clear all exceptions.
    ASSERT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) & e, 0);
  }

  ASSERT_EQ(LIBC_NAMESPACE::fesetround(FE_DOWNWARD), 0);
  ASSERT_EQ(LIBC_NAMESPACE::fesetenv(FE_DFL_ENV), 0);
  // Setting the default env should set rounding mode to FE_TONEAREST.
  int rm = LIBC_NAMESPACE::fegetround();
  EXPECT_EQ(rm, FE_TONEAREST);
}
#endif

#ifdef LIBC_TARGET_OS_IS_WINDOWS
TEST_F(LlvmLibcFEnvTest, Windows_Set_Get_Test) {
  // If a valid fenv_t is written, then reading it back out should be identical.
  fenv_t setEnv = {0x7e00053e, 0x0f00000f};
  fenv_t getEnv;
  ASSERT_EQ(LIBC_NAMESPACE::fesetenv(&setEnv), 0);
  ASSERT_EQ(LIBC_NAMESPACE::fegetenv(&getEnv), 0);

  ASSERT_EQ(setEnv._Fe_ctl, getEnv._Fe_ctl);
  ASSERT_EQ(setEnv._Fe_stat, getEnv._Fe_stat);
}
#endif
