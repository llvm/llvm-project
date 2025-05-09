//===-- Unittests for asin ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "src/math/asin.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAsinTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcAsinTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asin(sNaN),
                                           FE_INVALID);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asin(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::asin(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::asin(neg_zero));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asin(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asin(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asin(2.0));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asin(-2.0));
  EXPECT_FP_EQ(0x1.921fb54442d18p0, LIBC_NAMESPACE::asin(1.0));
  EXPECT_FP_EQ(-0x1.921fb54442d18p0, LIBC_NAMESPACE::asin(-1.0));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcAsinTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::asin(min_denormal));
}

TEST_F(LlvmLibcAsinTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::asin(min_denormal));
}

TEST_F(LlvmLibcAsinTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::asin(min_denormal));
}

#endif
