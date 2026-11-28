//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for expm1bf16.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/math/expm1bf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"

class LlvmLibcExpm1bf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)
public:
  void test_special_numbers() {
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::expm1bf16(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::expm1bf16(sNaN),
                                    FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::expm1bf16(inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(LIBC_NAMESPACE::fputil::cast<bfloat16>(-1.0f),
                              LIBC_NAMESPACE::expm1bf16(neg_inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::expm1bf16(zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::expm1bf16(neg_zero));
    EXPECT_MATH_ERRNO(0);
  }

  void test_overflow() {
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::expm1bf16(max_normal),
                                FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf,
        LIBC_NAMESPACE::expm1bf16(
            LIBC_NAMESPACE::fputil::cast<bfloat16>(89.0f)),
        FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);
  }

  void test_result_near_minus_one() {
    EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
        LIBC_NAMESPACE::fputil::cast<bfloat16>(-1.0f),
        LIBC_NAMESPACE::expm1bf16(neg_max_normal), FE_INEXACT);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
        LIBC_NAMESPACE::fputil::cast<bfloat16>(-1.0f),
        LIBC_NAMESPACE::expm1bf16(
            LIBC_NAMESPACE::fputil::cast<bfloat16>(-6.25f)),
        FE_INEXACT);
    EXPECT_MATH_ERRNO(0);
  }
};

TEST_F(LlvmLibcExpm1bf16Test, SpecialNumbers) { test_special_numbers(); }
TEST_F(LlvmLibcExpm1bf16Test, Overflow) { test_overflow(); }
TEST_F(LlvmLibcExpm1bf16Test, ResultNearMinusOne) {
  test_result_near_minus_one();
}
