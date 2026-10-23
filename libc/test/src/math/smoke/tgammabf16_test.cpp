//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for the tgammabf16 function.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/tgammabf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcTgammaBf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)

public:
  void test_special_numbers() {
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::tgammabf16(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::tgammabf16(sNaN),
                                FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::tgammabf16(inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::tgammabf16(neg_inf), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        inf, LIBC_NAMESPACE::tgammabf16(zero), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        neg_inf, LIBC_NAMESPACE::tgammabf16(neg_zero), FE_DIVBYZERO);
    EXPECT_MATH_ERRNO(ERANGE);
  }

  void test_poles() {
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::tgammabf16(bfloat16(-1.0f)), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::tgammabf16(bfloat16(-2.0f)), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);
  }

  void test_values() {
    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(1.0f),
                              LIBC_NAMESPACE::tgammabf16(bfloat16(1.0f)));
    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(1.0f),
                              LIBC_NAMESPACE::tgammabf16(bfloat16(2.0f)));
    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(2.0f),
                              LIBC_NAMESPACE::tgammabf16(bfloat16(3.0f)));
    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(24.0f),
                              LIBC_NAMESPACE::tgammabf16(bfloat16(5.0f)));
  }

  void test_boundaries() {
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::tgammabf16(bfloat16(36.0f)), FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);
  }
};

TEST_F(LlvmLibcTgammaBf16Test, SpecialNumbers) { test_special_numbers(); }
TEST_F(LlvmLibcTgammaBf16Test, Poles) { test_poles(); }
TEST_F(LlvmLibcTgammaBf16Test, Values) { test_values(); }
TEST_F(LlvmLibcTgammaBf16Test, Boundaries) { test_boundaries(); }
