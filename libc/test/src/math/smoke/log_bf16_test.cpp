//===-- Unittests for BFloat16 log(x) function =---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/macros/properties/types.h"
#include "src/math/log_bf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcLogBf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)

public:
  void test_special_numbers() {
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::log_bf16(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::log_bf16(sNaN),
                                FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::log_bf16(inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::log_bf16(neg_inf));
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        neg_inf, LIBC_NAMESPACE::log_bf16(zero), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        neg_inf, LIBC_NAMESPACE::log_bf16(neg_zero), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::log_bf16(bfloat16(1.0)));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::log_bf16(bfloat16(-1.0)));
    EXPECT_MATH_ERRNO(EDOM);
  }
};

TEST_F(LlvmLibcLogBf16Test, SpecialNumbers) { test_special_numbers(); }
