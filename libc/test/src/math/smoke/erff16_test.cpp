//===-- Unittests for erff16 ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/math/erff16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcErffTest = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcErffTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::erff16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::erff16(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f16, LIBC_NAMESPACE::erff16(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(-1.0f16, LIBC_NAMESPACE::erff16(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::erff16(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::erff16(neg_zero));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcErffTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);
  EXPECT_FP_EQ(0x1p-24f16, LIBC_NAMESPACE::erff16(min_denormal));
  EXPECT_FP_EQ(0x1.208p-14f16, LIBC_NAMESPACE::erff16(max_denormal));
}

TEST_F(LlvmLibcErffTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);
  EXPECT_FP_EQ(0x1p-24f16, LIBC_NAMESPACE::erff16(min_denormal));
  EXPECT_FP_EQ(0x1.208p-14f16, LIBC_NAMESPACE::erff16(max_denormal));
}

TEST_F(LlvmLibcErffTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0x1p-24f16, LIBC_NAMESPACE::erff16(min_denormal));
  EXPECT_FP_EQ(0x1.208p-14f16, LIBC_NAMESPACE::erff16(max_denormal));
}

#endif
