//===-- Unittests for log2f -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/log2f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcLog2fTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, __llvm_libc::log2f(aNaN));
  EXPECT_FP_EQ(inf, __llvm_libc::log2f(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::log2f(neg_inf), FE_INVALID);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, __llvm_libc::log2f(0.0f), FE_DIVBYZERO);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, __llvm_libc::log2f(-0.0f), FE_DIVBYZERO);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::log2f(-1.0f), FE_INVALID);
  EXPECT_FP_EQ(zero, __llvm_libc::log2f(1.0f));
}

TEST(LlvmLibcLog2fTest, TrickyInputs) {
  constexpr int N = 10;
  constexpr uint32_t INPUTS[N] = {
      0x3f7d57f5U, 0x3f7e3274U, 0x3f7ed848U, 0x3f7fd6ccU, 0x3f7fffffU,
      0x3f80079bU, 0x3f81d0b5U, 0x3f82e602U, 0x3f83c98dU, 0x3f8cba39U};

  for (int i = 0; i < N; ++i) {
    float x = float(FPBits(INPUTS[i]));
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log2, x,
                                   __llvm_libc::log2f(x), 0.5);
  }
}

TEST(LlvmLibcLog2fTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    errno = 0;
    float result = __llvm_libc::log2f(x);
    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || errno != 0)
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log2, x,
                                   __llvm_libc::log2f(x), 0.5);
  }
}
