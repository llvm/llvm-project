//===-- Unittests for lgammaf16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/math/lgammaf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcLgammaf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcLgammaf16Test, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::lgammaf16(sNaN),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::lgammaf16(aNaN));

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::lgammaf16(inf));

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::lgammaf16(neg_inf));

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf16(zero),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf16(neg_zero),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcLgammaf16Test, NegativeIntegers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf16(-1.0f16),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf16(-2.0f16),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf16(-3.0f16),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcLgammaf16Test, ExactValues) {
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::lgammaf16(1.0f16));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::lgammaf16(2.0f16));
}

TEST_F(LlvmLibcLgammaf16Test, Overflow) {
  // lgamma(8192) = 65623 > 65520 so it
  // overflows to +Inf in round-to-nearest mode.
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf16(8192.0f16),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);
}
