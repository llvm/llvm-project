//===-- Unittests for exp2m1f ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/exp2m1f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExp2m1fTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcExp2m1fTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::exp2m1f(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::exp2m1f(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(-1.0f, LIBC_NAMESPACE::exp2m1f(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::exp2m1f(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(-0.0f, LIBC_NAMESPACE::exp2m1f(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::exp2m1f(1.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(-0.5f, LIBC_NAMESPACE::exp2m1f(-1.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(3.0f, LIBC_NAMESPACE::exp2m1f(2.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(-0.75f, LIBC_NAMESPACE::exp2m1f(-2.0f));
}

TEST_F(LlvmLibcExp2m1fTest, Overflow) {
  LIBC_NAMESPACE::libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::exp2m1f(FPBits(0x7f7f'ffffU).get_val()),
      FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::exp2m1f(FPBits(0x4300'0000U).get_val()),
      FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::exp2m1f(FPBits(0x4300'0001U).get_val()),
      FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}
