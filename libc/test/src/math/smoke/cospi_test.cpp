//===-- Unittests for cospi -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/cast.h"
#include "src/errno/libc_errno.h"
#include "src/math/cospi.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcCospiTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcCospiTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cospi(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(FPBits::one().get_val(), LIBC_NAMESPACE::cospi(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(FPBits::one().get_val(), LIBC_NAMESPACE::cospi(neg_zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cospi(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cospi(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcCospiTest, Integers) {

  EXPECT_FP_EQ(FPBits::one(Sign::NEG).get_val(), LIBC_NAMESPACE::cospi(-0x1.8406003b2ae63p52));

  EXPECT_FP_EQ(FPBits::one().get_val(), LIBC_NAMESPACE::cospi(0x1p54));

  EXPECT_FP_EQ(FPBits::one().get_val(), LIBC_NAMESPACE::cospi(0x1p55));

  EXPECT_FP_EQ(FPBits::one().get_val(), LIBC_NAMESPACE::cospi(0x1p56));
}
