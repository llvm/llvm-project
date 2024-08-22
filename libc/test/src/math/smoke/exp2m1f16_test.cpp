//===-- Unittests for exp2m1f16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/errno/libc_errno.h"
#include "src/math/exp2m1f16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExp2m1f16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcExp2m1f16Test, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::exp2m1f16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::exp2m1f16(sNaN),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::exp2m1f16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(static_cast<float16>(-1.0),
                            LIBC_NAMESPACE::exp2m1f16(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::exp2m1f16(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::exp2m1f16(neg_zero));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExp2m1f16Test, Overflow) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::exp2m1f16(max_normal),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  float16 x = static_cast<float16>(16.0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      inf, LIBC_NAMESPACE::exp2m1f16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(inf, LIBC_NAMESPACE::exp2m1f16(x),
                                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      max_normal, LIBC_NAMESPACE::exp2m1f16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      max_normal, LIBC_NAMESPACE::exp2m1f16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExp2m1f16Test, ResultNearNegOne) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_WITH_EXCEPTION(static_cast<float16>(-1.0),
                              LIBC_NAMESPACE::exp2m1f16(neg_max_normal),
                              FE_INEXACT);

  EXPECT_FP_EQ_ALL_ROUNDING(
      static_cast<float16>(-0x1.ffcp-1),
      LIBC_NAMESPACE::exp2m1f16(static_cast<float16>(-11)));

  float16 x = static_cast<float16>(-12);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      static_cast<float16>(-1.0), LIBC_NAMESPACE::exp2m1f16(x), FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(static_cast<float16>(-0x1.ffcp-1),
                                              LIBC_NAMESPACE::exp2m1f16(x),
                                              FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      static_cast<float16>(-1.0), LIBC_NAMESPACE::exp2m1f16(x), FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      static_cast<float16>(-0x1.ffcp-1), LIBC_NAMESPACE::exp2m1f16(x),
      FE_INEXACT);
}
