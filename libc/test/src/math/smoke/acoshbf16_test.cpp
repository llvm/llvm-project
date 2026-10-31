//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for acoshbf16.
///
/// Verifies boundary values, special floating-point edge cases (NaN, sNaN,
/// positive/negative infinity, zero), out-of-domain inputs (x < 1.0), and
/// proper floating-point exception handling (FE_INVALID) along with errno
/// reporting (EDOM).
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/math/acoshbf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcAcoshBf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)
public:
  void test_special_numbers() {
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acoshbf16(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acoshbf16(sNaN), FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acoshbf16(zero), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acoshbf16(neg_zero), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::acoshbf16(inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acoshbf16(neg_inf), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::acoshbf16(bfloat16(1.0)));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acoshbf16(bfloat16(0.5)), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acoshbf16(bfloat16(-1.0)), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acoshbf16(bfloat16(-2.0)), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);
  }
};
TEST_F(LlvmLibcAcoshBf16Test, SpecialNumbers) { test_special_numbers(); }
