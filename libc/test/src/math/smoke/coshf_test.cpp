//===-- Unittests for coshf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/coshf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using FPBits = LIBC_NAMESPACE::fputil::FPBits<float>;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcCoshfTest, SpecialNumbers) {
  libc_errno = 0;

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::coshf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::coshf(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::coshf(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::coshf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::coshf(-0.0f));
  EXPECT_MATH_ERRNO(0);
}

TEST(LlvmLibcCoshfTest, Overflow) {
  libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::coshf(float(FPBits(0x7f7fffffU))), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::coshf(float(FPBits(0x42cffff8U))), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::coshf(float(FPBits(0x42d00008U))), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}
