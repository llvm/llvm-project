//===-- Unittests for BFloat16 log10(1 + x) function ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/macros/properties/types.h"
#include "src/math/log10p1bf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using FPBits = LIBC_NAMESPACE::fputil::FPBits<bfloat16>;

class LlvmLibcLog10p1bf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)

public:
  void test_special_numbers() {
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::log10p1bf16(aNaN), 0);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::log10p1bf16(sNaN), FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        inf, LIBC_NAMESPACE::log10p1bf16(inf), 0);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::log10p1bf16(neg_inf), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        zero, LIBC_NAMESPACE::log10p1bf16(zero), 0);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        neg_zero, LIBC_NAMESPACE::log10p1bf16(neg_zero), 0);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        neg_inf, LIBC_NAMESPACE::log10p1bf16(bfloat16(-1.0)), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(0);

    const bfloat16 neg_two = FPBits(static_cast<uint16_t>(0xc000U)).get_val();
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::log10p1bf16(neg_two), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    bfloat16 neg_one_adjacent =
        FPBits(static_cast<uint16_t>(0xbf81U)).get_val();
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::log10p1bf16(neg_one_adjacent), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    bfloat16 neg_inf_adjacent =
        FPBits(static_cast<uint16_t>(0xff7fU)).get_val();
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::log10p1bf16(neg_inf_adjacent), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    const bfloat16 one = bfloat16(1.0);
    const bfloat16 nine = bfloat16(9.0);
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        one, LIBC_NAMESPACE::log10p1bf16(nine), 0);
    EXPECT_MATH_ERRNO(0);

    const bfloat16 two = bfloat16(2.0);
    const bfloat16 ninety_nine = bfloat16(99.0);
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        two, LIBC_NAMESPACE::log10p1bf16(ninety_nine), 0);
    EXPECT_MATH_ERRNO(0);
  }
};

TEST_F(LlvmLibcLog10p1bf16Test, SpecialNumbers) { test_special_numbers(); }
