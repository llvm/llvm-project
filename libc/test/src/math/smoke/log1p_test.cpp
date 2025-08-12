//===-- Unittests for log1p -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/math/log1p.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcLog1pTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcLog1pTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::log1p(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::log1p(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::log1p(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log1p(neg_inf), FE_INVALID);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log1p(-2.0), FE_INVALID);
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::log1p(0.0));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::log1p(-0.0));
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log1p(-1.0),
                              FE_DIVBYZERO);

  EXPECT_FP_EQ(0x1.62c829bf8fd9dp9,
               LIBC_NAMESPACE::log1p(0x1.9b536cac3a09dp1023));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcLog1pTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::log1p(min_denormal));
  EXPECT_FP_EQ(0x1.62c829bf8fd9dp9,
               LIBC_NAMESPACE::log1p(0x1.9b536cac3a09dp1023));
}

TEST_F(LlvmLibcLog1pTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::log1p(min_denormal));
  EXPECT_FP_EQ(0x1.62c829bf8fd9dp9,
               LIBC_NAMESPACE::log1p(0x1.9b536cac3a09dp1023));
}

TEST_F(LlvmLibcLog1pTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::log1p(min_denormal));
  EXPECT_FP_EQ(0x1.62c829bf8fd9dp9,
               LIBC_NAMESPACE::log1p(0x1.9b536cac3a09dp1023));
}

#endif
