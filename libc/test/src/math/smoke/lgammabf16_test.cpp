//===-- Unittests for lgammabf16 ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/lgammabf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcLgammaBf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)

public:
  void test_special_numbers() {
    // aNaN -> aNaN, no exception
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::lgammabf16(aNaN));
    EXPECT_MATH_ERRNO(0);

    // sNaN -> aNaN, FE_INVALID
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::lgammabf16(sNaN),
                                FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    // +Inf -> +Inf
    EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::lgammabf16(inf));
    EXPECT_MATH_ERRNO(0);

    // -Inf -> +Inf
    EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::lgammabf16(neg_inf));
    EXPECT_MATH_ERRNO(0);

    // +-0 -> +Inf, pole error
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        inf, LIBC_NAMESPACE::lgammabf16(zero), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        inf, LIBC_NAMESPACE::lgammabf16(neg_zero), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(ERANGE);

    // lgamma(1) = lgamma(2) = 0
    EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::lgammabf16(bfloat16(1.0f)));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::lgammabf16(bfloat16(2.0f)));
    EXPECT_MATH_ERRNO(0);

    // Negative integers -> +Inf, pole error
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        inf, LIBC_NAMESPACE::lgammabf16(bfloat16(-1.0f)), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        inf, LIBC_NAMESPACE::lgammabf16(bfloat16(-2.0f)), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(ERANGE);
  }
  void test_negative_integers() {
    // More negative integer poles -> +Inf, pole error
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        inf, LIBC_NAMESPACE::lgammabf16(bfloat16(-3.0f)), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(ERANGE);
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        inf, LIBC_NAMESPACE::lgammabf16(bfloat16(-4.0f)), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(ERANGE);
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        inf, LIBC_NAMESPACE::lgammabf16(bfloat16(-10.0f)), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(ERANGE);
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        inf, LIBC_NAMESPACE::lgammabf16(bfloat16(-100.0f)), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(ERANGE);
  }

  void test_exact_values() {
    // lgamma(3) = ln(2) ~ 0.693147
    EXPECT_FP_EQ(bfloat16(0x1.62p-1f),
                 LIBC_NAMESPACE::lgammabf16(bfloat16(3.0f)));
    EXPECT_MATH_ERRNO(0);
    // lgamma(4) = ln(6) ~ 1.791759
    EXPECT_FP_EQ(bfloat16(0x1.cap+0f),
                 LIBC_NAMESPACE::lgammabf16(bfloat16(4.0f)));
    EXPECT_MATH_ERRNO(0);
    // lgamma(0.5) = ln(sqrt(pi)) ~ 0.572365
    EXPECT_FP_EQ(bfloat16(0x1.26p-1f),
                 LIBC_NAMESPACE::lgammabf16(bfloat16(0.5f)));
    EXPECT_MATH_ERRNO(0);
  }

  void test_overflow() {
    // Large x -> lgamma overflows to +Inf
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::lgammabf16(max_normal),
                                FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);
  }
};

TEST_F(LlvmLibcLgammaBf16Test, SpecialNumbers) { test_special_numbers(); }
TEST_F(LlvmLibcLgammaBf16Test, NegativeIntegers) { test_negative_integers(); }
TEST_F(LlvmLibcLgammaBf16Test, ExactValues) { test_exact_values(); }
TEST_F(LlvmLibcLgammaBf16Test, Overflow) { test_overflow(); }
