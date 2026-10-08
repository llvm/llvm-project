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
#include "test/UnitTest/RoundingModeUtils.h"
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
  // lgamma(x) overflows float around x >= 0x1.895f1cp+121f (~ 2^121).
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf(0x1.895f1cp+121f),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf(0x1.896p+121f),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf(0x1.898p+121f),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammaf(0x1p126f),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
  using LIBC_NAMESPACE::fputil::testing::RoundingMode;

  const RoundingMode modes[] = {RoundingMode::Nearest, RoundingMode::Upward,
                                RoundingMode::Downward,
                                RoundingMode::TowardZero};
  const float overflow_inputs[] = {0x1.895f1cp+121f, 0x1.896p+121f,
                                   0x1.898p+121f, 0x1p126f};

  for (RoundingMode mode : modes) {
    ForceRoundingMode r(mode);
    if (!r.success)
      continue;
    for (float x : overflow_inputs) {
      libc_errno = 0;
      float expected =
          (mode == RoundingMode::Downward || mode == RoundingMode::TowardZero)
              ? FPBits::max_normal().get_val()
              : inf;
      EXPECT_FP_EQ_WITH_EXCEPTION(expected, LIBC_NAMESPACE::lgammaf(x),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }
  }
}
