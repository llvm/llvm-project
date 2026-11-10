//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for bfloat16 cosh.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/coshbf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcCoshBf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)
public:
  void test_special_numbers() {
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::coshbf16(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::coshbf16(sNaN),
                                FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::coshbf16(inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::coshbf16(neg_inf));
    EXPECT_MATH_ERRNO(0);

    // cosh(+-0) = 1 exactly
    bfloat16 one = FPBits(uint16_t(0x3F80)).get_val(); // 0x1p0
    EXPECT_FP_EQ_ALL_ROUNDING(one, LIBC_NAMESPACE::coshbf16(zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(one, LIBC_NAMESPACE::coshbf16(neg_zero));
    EXPECT_MATH_ERRNO(0);
  }

  // For 0 < |x| <= 0.0879, cosh(x) is strictly between 1 and the next bfloat16
  // above 1, and is closer to 1. So the result is 1 in all rounding modes
  // except for RU, where it is the next value above 1.
  void test_small_values() {
    bfloat16 one = FPBits(uint16_t(0x3F80)).get_val();         // 0x1p0
    bfloat16 one_next_up = FPBits(uint16_t(0x3F81)).get_val(); // 0x1.02p0

    // Largest |x| for which cosh(x) still rounds to 1 in RN.
    bfloat16 rounds_to_one_bound =
        FPBits(uint16_t(0x3DB4)).get_val(); // 0x1.68p-4
    // Smallest |x| for which cosh(x) rounds to the next value above 1 in RN.
    bfloat16 rounds_above_one_bound =
        FPBits(uint16_t(0x3DB5)).get_val(); // 0x1.6ap-4

    EXPECT_FP_EQ_ROUNDING_NEAREST(one, LIBC_NAMESPACE::coshbf16(min_denormal));
    EXPECT_FP_EQ_ROUNDING_UPWARD(one_next_up,
                                 LIBC_NAMESPACE::coshbf16(min_denormal));
    EXPECT_FP_EQ_ROUNDING_DOWNWARD(one, LIBC_NAMESPACE::coshbf16(min_denormal));
    EXPECT_FP_EQ_ROUNDING_TOWARD_ZERO(one,
                                      LIBC_NAMESPACE::coshbf16(min_denormal));

    EXPECT_FP_EQ_ROUNDING_NEAREST(one,
                                  LIBC_NAMESPACE::coshbf16(neg_min_denormal));
    EXPECT_FP_EQ_ROUNDING_NEAREST(
        one, LIBC_NAMESPACE::coshbf16(rounds_to_one_bound));
    EXPECT_FP_EQ_ROUNDING_NEAREST(
        one, LIBC_NAMESPACE::coshbf16(-rounds_to_one_bound));
    EXPECT_FP_EQ_ROUNDING_UPWARD(one_next_up,
                                 LIBC_NAMESPACE::coshbf16(rounds_to_one_bound));

    EXPECT_FP_EQ_ROUNDING_NEAREST(
        one_next_up, LIBC_NAMESPACE::coshbf16(rounds_above_one_bound));
    EXPECT_FP_EQ_ROUNDING_NEAREST(
        one_next_up, LIBC_NAMESPACE::coshbf16(-rounds_above_one_bound));
  }

  // cosh(x) overflows bfloat16 for |x| >= 89.5. cosh(x) > 0 for all x, so the
  // overflow is always to +Inf, regardless of the sign of x.
  void test_overflow() {
    bfloat16 overflow_bound = FPBits(uint16_t(0x42B3)).get_val(); // 0x1.66p6

    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::coshbf16(overflow_bound),
                                FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::coshbf16(-overflow_bound),
                                FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::coshbf16(max_normal),
                                FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::coshbf16(neg_max_normal),
                                FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);
  }
};

TEST_F(LlvmLibcCoshBf16Test, SpecialNumbers) { test_special_numbers(); }
TEST_F(LlvmLibcCoshBf16Test, SmallValues) { test_small_values(); }
TEST_F(LlvmLibcCoshBf16Test, Overflow) { test_overflow(); }
