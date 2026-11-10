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

    // cosh(+-0) = 1 exactly.
    bfloat16 one = FPBits::one().get_val();
    EXPECT_FP_EQ_ALL_ROUNDING(one, LIBC_NAMESPACE::coshbf16(zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(one, LIBC_NAMESPACE::coshbf16(neg_zero));
    EXPECT_MATH_ERRNO(0);
  }

  // For small values, cosh(x) rounds to 1 (or to the next value above 1 when
  // rounding upward, since cosh(x) > 1 for x != 0).
  void test_small_values() {
    bfloat16 one = FPBits::one().get_val();
    bfloat16 one_plus_ulp = FPBits(uint16_t(0x3f81)).get_val();

    // Largest inputs (in magnitude) that still round to 1, and the smallest
    // subnormal.
    EXPECT_FP_EQ(one,
                 LIBC_NAMESPACE::coshbf16(FPBits(uint16_t(0x0001)).get_val()));
    EXPECT_FP_EQ(one,
                 LIBC_NAMESPACE::coshbf16(FPBits(uint16_t(0x8001)).get_val()));
    EXPECT_FP_EQ(one,
                 LIBC_NAMESPACE::coshbf16(FPBits(uint16_t(0x3db4)).get_val()));
    EXPECT_FP_EQ(one,
                 LIBC_NAMESPACE::coshbf16(FPBits(uint16_t(0xbdb4)).get_val()));

    // First input whose result no longer rounds to 1 in round-to-nearest.
    EXPECT_FP_EQ(one_plus_ulp,
                 LIBC_NAMESPACE::coshbf16(FPBits(uint16_t(0x3db5)).get_val()));
    EXPECT_FP_EQ(one_plus_ulp,
                 LIBC_NAMESPACE::coshbf16(FPBits(uint16_t(0xbdb5)).get_val()));
  }

  void test_overflow() {
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::coshbf16(FPBits(uint16_t(0x42b3)).get_val()),
        FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::coshbf16(max_normal),
                                FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    // cosh is even: large negative inputs overflow to +inf as well.
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::coshbf16(FPBits(uint16_t(0xc2b3)).get_val()),
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
