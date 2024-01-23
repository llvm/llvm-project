//===-- Unittests for expm1f-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/expm1f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <stdint.h>

using LlvmLibcExpm1fTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcExpm1fTest, SpecialNumbers) {
  libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::expm1f(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::expm1f(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-1.0f, LIBC_NAMESPACE::expm1f(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::expm1f(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-0.0f, LIBC_NAMESPACE::expm1f(-0.0f));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExpm1fTest, Overflow) {
  libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::expm1f(FPBits(0x7f7fffffU).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::expm1f(FPBits(0x42cffff8U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::expm1f(FPBits(0x42d00008U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcExpm1fTest, Underflow) {
  libc_errno = 0;
  EXPECT_FP_EQ(-1.0f, LIBC_NAMESPACE::expm1f(FPBits(0xff7fffffU).get_val()));

  float x = FPBits(0xc2cffff8U).get_val();
  EXPECT_FP_EQ(-1.0f, LIBC_NAMESPACE::expm1f(x));

  x = FPBits(0xc2d00008U).get_val();
  EXPECT_FP_EQ(-1.0f, LIBC_NAMESPACE::expm1f(x));
}

// Test with inputs which are the borders of underflow/overflow but still
// produce valid results without setting errno.
TEST_F(LlvmLibcExpm1fTest, Borderline) {
  float x;

  libc_errno = 0;
  x = FPBits(0x42affff8U).get_val();
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Expm1, x,
                                 LIBC_NAMESPACE::expm1f(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = FPBits(0x42b00008U).get_val();
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Expm1, x,
                                 LIBC_NAMESPACE::expm1f(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = FPBits(0xc2affff8U).get_val();
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Expm1, x,
                                 LIBC_NAMESPACE::expm1f(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = FPBits(0xc2b00008U).get_val();
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Expm1, x,
                                 LIBC_NAMESPACE::expm1f(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = FPBits(0x3dc252ddU).get_val();
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Expm1, x,
                                 LIBC_NAMESPACE::expm1f(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = FPBits(0x3e35bec5U).get_val();
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Expm1, x,
                                 LIBC_NAMESPACE::expm1f(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = FPBits(0x942ed494U).get_val();
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Expm1, x,
                                 LIBC_NAMESPACE::expm1f(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = FPBits(0xbdc1c6cbU).get_val();
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Expm1, x,
                                 LIBC_NAMESPACE::expm1f(x), 0.5);
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExpm1fTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (isnan(x) || isinf(x))
      continue;
    libc_errno = 0;
    float result = LIBC_NAMESPACE::expm1f(x);

    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || libc_errno != 0)
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Expm1, x,
                                   LIBC_NAMESPACE::expm1f(x), 0.5);
  }
}
