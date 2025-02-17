//===-- Unittests for logf16 ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/cast.h"
#include "src/errno/libc_errno.h"
#include "src/math/logf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcLogf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcLogf16Test, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::logf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::logf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::logf16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::logf16(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
      neg_inf, LIBC_NAMESPACE::logf16(zero), FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
      neg_inf, LIBC_NAMESPACE::logf16(neg_zero), FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(
      zero, LIBC_NAMESPACE::logf16(LIBC_NAMESPACE::fputil::cast<float16>(1.0)));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(
      aNaN,
      LIBC_NAMESPACE::logf16(LIBC_NAMESPACE::fputil::cast<float16>(-1.0)));
  EXPECT_MATH_ERRNO(EDOM);
}
