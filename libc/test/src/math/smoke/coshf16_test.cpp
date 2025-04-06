//===-- Unittests for coshf16 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/cast.h"
#include "src/errno/libc_errno.h"
#include "src/math/coshf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcCoshf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcCoshf16Test, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::coshf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::coshf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::coshf16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::coshf16(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(LIBC_NAMESPACE::fputil::cast<float16>(1.0),
                            LIBC_NAMESPACE::coshf16(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(LIBC_NAMESPACE::fputil::cast<float16>(1.0),
                            LIBC_NAMESPACE::coshf16(neg_zero));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcCoshf16Test, Overflow) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::coshf16(max_normal),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::coshf16(neg_max_normal),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  // round(acosh(2^16), HP, RU);
  float16 x = LIBC_NAMESPACE::fputil::cast<float16>(0x1.794p+3);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(inf, LIBC_NAMESPACE::coshf16(x),
                                               FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(inf, LIBC_NAMESPACE::coshf16(x),
                                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      max_normal, LIBC_NAMESPACE::coshf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      max_normal, LIBC_NAMESPACE::coshf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  // round(-acosh(2^16), HP, RD);
  x = LIBC_NAMESPACE::fputil::cast<float16>(-0x1.794p+3);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(inf, LIBC_NAMESPACE::coshf16(x),
                                               FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(inf, LIBC_NAMESPACE::coshf16(x),
                                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      max_normal, LIBC_NAMESPACE::coshf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      max_normal, LIBC_NAMESPACE::coshf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
}
