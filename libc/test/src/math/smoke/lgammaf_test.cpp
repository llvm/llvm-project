//===-- Unittests for lgammaf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/math/lgammaf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcLgammafTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcLgammafTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::lgammaf(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::lgammaf(aNaN));

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::lgammaf(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::lgammaf(neg_inf));

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf(zero), FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf(neg_zero),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcLgammafTest, NegativeIntegers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf(-1.0f),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf(-2.0f),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf(-100.0f),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  // Large negative integer (still representable as float).
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf(-0x1p23f),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcLgammafTest, ExactValues) {
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::lgammaf(1.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::lgammaf(2.0f));
}

TEST_F(LlvmLibcLgammafTest, Overflow) {
  // lgamma(x) overflows float around x ~ 2^121. Pick a comfortable margin.
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf(0x1p126f),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);
}
