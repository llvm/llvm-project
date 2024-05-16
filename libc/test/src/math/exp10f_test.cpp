//===-- Unittests for exp10f ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/exp10f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <stdint.h>

using LlvmLibcExp10fTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcExp10fTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::exp10f(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::exp10f(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::exp10f(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::exp10f(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::exp10f(-0.0f));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExp10fTest, Overflow) {
  LIBC_NAMESPACE::libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::exp10f(FPBits(0x7f7fffffU).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::exp10f(FPBits(0x43000000U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::exp10f(FPBits(0x43000001U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcExp10fTest, Underflow) {
  LIBC_NAMESPACE::libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION(
      0.0f, LIBC_NAMESPACE::exp10f(FPBits(0xff7fffffU).get_val()),
      FE_UNDERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  float x = FPBits(0xc2cffff8U).get_val();
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10, x,
                                 LIBC_NAMESPACE::exp10f(x), 0.5);
  EXPECT_MATH_ERRNO(ERANGE);

  x = FPBits(0xc2d00008U).get_val();
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10, x,
                                 LIBC_NAMESPACE::exp10f(x), 0.5);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcExp10fTest, TrickyInputs) {
  constexpr int N = 20;
  constexpr uint32_t INPUTS[N] = {
      0x325e5bd8, // x = 0x1.bcb7bp-27f
      0x325e5bd9, // x = 0x1.bcb7b2p-27f
      0x325e5bda, // x = 0x1.bcb7b4p-27f
      0x3d14d956, // x = 0x1.29b2acp-5f
      0x4116498a, // x = 0x1.2c9314p3f
      0x4126f431, // x = 0x1.4de862p3f
      0x4187d13c, // x = 0x1.0fa278p4f
      0x4203e9da, // x = 0x1.07d3b4p5f
      0x420b5f5d, // x = 0x1.16bebap5f
      0x42349e35, // x = 0x1.693c6ap5f
      0x3f800000, // x = 1.0f
      0x40000000, // x = 2.0f
      0x40400000, // x = 3.0f
      0x40800000, // x = 4.0f
      0x40a00000, // x = 5.0f
      0x40c00000, // x = 6.0f
      0x40e00000, // x = 7.0f
      0x41000000, // x = 8.0f
      0x41100000, // x = 9.0f
      0x41200000, // x = 10.0f
  };
  for (int i = 0; i < N; ++i) {
    LIBC_NAMESPACE::libc_errno = 0;
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10, x,
                                   LIBC_NAMESPACE::exp10f(x), 0.5);
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10, -x,
                                   LIBC_NAMESPACE::exp10f(-x), 0.5);
  }
}

TEST_F(LlvmLibcExp10fTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (isnan(x) || isinf(x))
      continue;
    LIBC_NAMESPACE::libc_errno = 0;
    float result = LIBC_NAMESPACE::exp10f(x);

    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || LIBC_NAMESPACE::libc_errno != 0)
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10, x,
                                   LIBC_NAMESPACE::exp10f(x), 0.5);
  }
}
