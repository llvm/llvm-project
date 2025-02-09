//===-- Unittests for cos -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cos.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcCosTest = LIBC_NAMESPACE::testing::FPTest<double>;

using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcCosTest, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::cos(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::cos(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::cos(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, LIBC_NAMESPACE::cos(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, LIBC_NAMESPACE::cos(neg_zero));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::cos(0x1.0p-50));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::cos(min_normal));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::cos(min_denormal));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcCosTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::cos(min_denormal));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::cos(max_denormal));
}

TEST_F(LlvmLibcCosTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::cos(min_denormal));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::cos(max_denormal));
}

TEST_F(LlvmLibcCosTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::cos(min_denormal));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::cos(max_denormal));
}

#endif
