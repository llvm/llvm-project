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
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using LlvmLibcCoshfTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcCoshfTest, SpecialNumbers) {
  libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::coshf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::coshf(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::coshf(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::coshf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::coshf(-0.0f));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcCoshfTest, Overflow) {
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

TEST_F(LlvmLibcCoshfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Cosh, x, LIBC_NAMESPACE::coshf(x), 0.5);
  }
}

TEST_F(LlvmLibcCoshfTest, SmallValues) {
  float x = float(FPBits(0x17800000U));
  float result = LIBC_NAMESPACE::coshf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cosh, x, result, 0.5);
  EXPECT_FP_EQ(1.0f, result);

  x = float(FPBits(0x0040000U));
  result = LIBC_NAMESPACE::coshf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cosh, x, result, 0.5);
  EXPECT_FP_EQ(1.0f, result);
}
