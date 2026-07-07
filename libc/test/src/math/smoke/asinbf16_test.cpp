//===-- Unittests for asinbf16 --------------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/asinbf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcAsinBf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)
public:
  void test_special_numbers() {
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asinbf16(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::asinbf16(sNaN), FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::asinbf16(zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::asinbf16(neg_zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asinbf16(inf));
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asinbf16(neg_inf));
    EXPECT_MATH_ERRNO(EDOM);
  }
};
TEST_F(LlvmLibcAsinBf16Test, SpecialNumbers) { test_special_numbers(); }
