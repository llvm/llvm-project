//===-- Unittests for acosbf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/acosbf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcAcosBf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)
public:
  void test_special_numbers() {
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acosbf16(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acosbf16(sNaN), FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acosbf16(inf), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acosbf16(neg_inf), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(0x1.921fb6p0f),
                              LIBC_NAMESPACE::acosbf16(zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(0x1.921fb6p0f),
                              LIBC_NAMESPACE::acosbf16(neg_zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::acosbf16(bfloat16(1.0)));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(0x1.921fb6p1f),
                              LIBC_NAMESPACE::acosbf16(bfloat16(-1.0)));
    EXPECT_MATH_ERRNO(0);
  }
};
TEST_F(LlvmLibcAcosBf16Test, SpecialNumbers) { test_special_numbers(); }
