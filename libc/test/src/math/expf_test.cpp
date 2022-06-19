//===-- Unittests for expf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/expf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcExpfTest, SpecialNumbers) {
  errno = 0;

  EXPECT_FP_EQ(aNaN, __llvm_libc::expf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, __llvm_libc::expf(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, __llvm_libc::expf(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, __llvm_libc::expf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, __llvm_libc::expf(-0.0f));
  EXPECT_MATH_ERRNO(0);
}

TEST(LlvmLibcExpfTest, Overflow) {
  errno = 0;
  EXPECT_FP_EQ(inf, __llvm_libc::expf(float(FPBits(0x7f7fffffU))));
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ(inf, __llvm_libc::expf(float(FPBits(0x42cffff8U))));
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ(inf, __llvm_libc::expf(float(FPBits(0x42d00008U))));
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST(LlvmLibcExpfTest, Underflow) {
  errno = 0;
  EXPECT_FP_EQ(0.0f, __llvm_libc::expf(float(FPBits(0xff7fffffU))));
  EXPECT_MATH_ERRNO(ERANGE);

  float x = float(FPBits(0xc2cffff8U));
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, __llvm_libc::expf(x),
                                 0.5);
  EXPECT_MATH_ERRNO(ERANGE);

  x = float(FPBits(0xc2d00008U));
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, __llvm_libc::expf(x),
                                 0.5);
  EXPECT_MATH_ERRNO(ERANGE);
}

// Test with inputs which are the borders of underflow/overflow but still
// produce valid results without setting errno.
TEST(LlvmLibcExpfTest, Borderline) {
  float x;

  errno = 0;
  x = float(FPBits(0x42affff8U));
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, __llvm_libc::expf(x),
                                 0.5);
  EXPECT_MATH_ERRNO(0);

  x = float(FPBits(0x42b00008U));
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, __llvm_libc::expf(x),
                                 0.5);
  EXPECT_MATH_ERRNO(0);

  x = float(FPBits(0xc2affff8U));
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, __llvm_libc::expf(x),
                                 0.5);
  EXPECT_MATH_ERRNO(0);

  x = float(FPBits(0xc2b00008U));
  ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, __llvm_libc::expf(x),
                                 0.5);
  EXPECT_MATH_ERRNO(0);

  x = float(FPBits(0xc236bd8cU));
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, __llvm_libc::expf(x),
                                 0.5);
  EXPECT_MATH_ERRNO(0);
}

TEST(LlvmLibcExpfTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    errno = 0;
    float result = __llvm_libc::expf(x);

    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || errno != 0)
      continue;
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x,
                                   __llvm_libc::expf(x), 0.5);
  }
}
