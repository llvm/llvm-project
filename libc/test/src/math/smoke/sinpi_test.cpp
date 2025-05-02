//===-- Unittests for sinpi -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/cast.h"
#include "src/errno/libc_errno.h"
#include "src/math/sinpi.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcSinpiTest = LIBC_NAMESPACE::testing::FPTest<double>;
/*
TEST_F(LlvmLibcSinpiTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinpi(aNaN));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::sinpi(zero));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::sinpi(neg_zero));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinpi(inf));
  EXPECT_MATH_ERRNO(EDOM);
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinpi(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}
TEST_F(LlvmLibcSinpiTest, Integers) {
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::sinpi(-0x1.0000000000003p52));

  ASSERT_FP_EQ(-1.0, LIBC_NAMESPACE::sinpi(4499003037990983.5));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::sinpi(-0x1.0000000000005p52));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::sinpi(-0x1.0000000000006p52));
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::sinpi(0x1.0p1020));
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::sinpi(0x1.0000000000003p52));
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::sinpi(0x1.0000000000005p52));
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::sinpi(0x1.0000000000006p52));
}
*/

TEST_F(LlvmLibcSinpiTest, Integers) {
  // ASSERT_FP_EQ(zero,
  // LIBC_NAMESPACE::sinpi(4503563146482784.0));

  ASSERT_FP_EQ(
      -1.0,
      LIBC_NAMESPACE::sinpi(
			    4499003037990983.5));
}
