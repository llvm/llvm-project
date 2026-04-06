//===-- Unittests for asinpi ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "src/math/asinpi.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAsinpiTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcAsinpiTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asinpi(sNaN),
                                           FE_INVALID);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asinpi(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::asinpi(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::asinpi(neg_zero));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asinpi(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asinpi(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asinpi(2.0));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asinpi(-2.0));
  EXPECT_FP_EQ(0.5, LIBC_NAMESPACE::asinpi(1.0));
  EXPECT_FP_EQ(-0.5, LIBC_NAMESPACE::asinpi(-1.0));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcAsinpiTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_TRUE(zero == LIBC_NAMESPACE::asinpi(min_denormal));
}

TEST_F(LlvmLibcAsinpiTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_TRUE(zero == LIBC_NAMESPACE::asinpi(min_denormal));
}

TEST_F(LlvmLibcAsinpiTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_TRUE(zero == LIBC_NAMESPACE::asinpi(min_denormal));
}

#endif
