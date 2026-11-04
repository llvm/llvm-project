//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for bfloat16 sinh.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/sinhbf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcSinhBf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)
public:
  void test_special_numbers() {
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::sinhbf16(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::sinhbf16(sNaN),
                                FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::sinhbf16(inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, LIBC_NAMESPACE::sinhbf16(neg_inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::sinhbf16(zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::sinhbf16(neg_zero));
    EXPECT_MATH_ERRNO(0);
  }

  // For small values, sinh(x) is x.
  void test_small_values() {
    bfloat16 x = FPBits(uint16_t(0x3de8)).get_val();
    bfloat16 result = LIBC_NAMESPACE::sinhbf16(x);
    EXPECT_FP_EQ(x, result);

    x = FPBits(uint16_t(0xbde8)).get_val();
    result = LIBC_NAMESPACE::sinhbf16(x);
    EXPECT_FP_EQ(x, result);

    x = FPBits(uint16_t(0x0001)).get_val();
    result = LIBC_NAMESPACE::sinhbf16(x);
    EXPECT_FP_EQ(x, result);

    x = FPBits(uint16_t(0x8001)).get_val();
    result = LIBC_NAMESPACE::sinhbf16(x);
    EXPECT_FP_EQ(x, result);
  }

  void test_overflow() {
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::sinhbf16(FPBits(uint16_t(0x42b3)).get_val()),
        FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::sinhbf16(max_normal),
                                FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION(
        neg_inf, LIBC_NAMESPACE::sinhbf16(FPBits(uint16_t(0xc2b3)).get_val()),
        FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION(
        neg_inf, LIBC_NAMESPACE::sinhbf16(neg_max_normal), FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);
  }
};

TEST_F(LlvmLibcSinhBf16Test, SpecialNumbers) { test_special_numbers(); }
TEST_F(LlvmLibcSinhBf16Test, SmallValues) { test_small_values(); }
TEST_F(LlvmLibcSinhBf16Test, Overflow) { test_overflow(); }
