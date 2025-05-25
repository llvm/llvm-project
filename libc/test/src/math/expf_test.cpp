//===-- Unittests for expf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/expf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <stdint.h>

using LlvmLibcExpfTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcExpfTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::expf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::expf(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::expf(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::expf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::expf(-0.0f));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExpfTest, Overflow) {
  LIBC_NAMESPACE::libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::expf(FPBits(0x7f7fffffU).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::expf(FPBits(0x42cffff8U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::expf(FPBits(0x42d00008U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcExpfTest, Underflow) {
  LIBC_NAMESPACE::libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION(
      0.0f, LIBC_NAMESPACE::expf(FPBits(0xff7fffffU).get_val()), FE_UNDERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  float x = FPBits(0xc2cffff8U).get_val();
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x,
                                 LIBC_NAMESPACE::expf(x), 0.5);
  EXPECT_MATH_ERRNO(ERANGE);

  x = FPBits(0xc2d00008U).get_val();
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x,
                                 LIBC_NAMESPACE::expf(x), 0.5);
  EXPECT_MATH_ERRNO(ERANGE);
}

// Test with inputs which are the borders of underflow/overflow but still
// produce valid results without setting errno.
TEST_F(LlvmLibcExpfTest, Borderline) {
  float x;

  LIBC_NAMESPACE::libc_errno = 0;
  x = FPBits(0x42affff8U).get_val();
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x,
                                 LIBC_NAMESPACE::expf(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = FPBits(0x42b00008U).get_val();
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x,
                                 LIBC_NAMESPACE::expf(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = FPBits(0xc2affff8U).get_val();
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x,
                                 LIBC_NAMESPACE::expf(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = FPBits(0xc2b00008U).get_val();
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x,
                                 LIBC_NAMESPACE::expf(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = FPBits(0xc236bd8cU).get_val();
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x,
                                 LIBC_NAMESPACE::expf(x), 0.5);
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExpfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (FPBits(v).is_nan() || FPBits(v).is_inf())
      continue;
    LIBC_NAMESPACE::libc_errno = 0;
    float result = LIBC_NAMESPACE::expf(x);

    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (FPBits(result).is_nan() || FPBits(result).is_inf() ||
        LIBC_NAMESPACE::libc_errno != 0)
      continue;
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x,
                                   LIBC_NAMESPACE::expf(x), 0.5);
  }
}
