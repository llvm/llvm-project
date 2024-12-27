//===-- Unittests for tan -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/tan.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcTanTest = LIBC_NAMESPACE::testing::FPTest<double>;

using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcTanTest, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::tan(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::tan(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::tan(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::tan(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::tan(neg_zero));
  EXPECT_FP_EQ(0x1.0p-50, LIBC_NAMESPACE::tan(0x1.0p-50));
  EXPECT_FP_EQ(min_normal, LIBC_NAMESPACE::tan(min_normal));
  EXPECT_FP_EQ(min_denormal, LIBC_NAMESPACE::tan(min_denormal));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcTanTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::tan(min_denormal));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::tan(max_denormal));
}

TEST_F(LlvmLibcTanTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::tan(min_denormal));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::tan(max_denormal));
}

TEST_F(LlvmLibcTanTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::tan(min_denormal));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::tan(max_denormal));
}

#endif
