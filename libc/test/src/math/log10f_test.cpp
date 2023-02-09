//===-- Unittests for log10f ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/log10f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcLog10fTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, __llvm_libc::log10f(aNaN));
  EXPECT_FP_EQ(inf, __llvm_libc::log10f(inf));
  EXPECT_TRUE(FPBits(__llvm_libc::log10f(neg_inf)).is_nan());
  EXPECT_FP_EQ(neg_inf, __llvm_libc::log10f(0.0f));
  EXPECT_FP_EQ(neg_inf, __llvm_libc::log10f(-0.0f));
  EXPECT_TRUE(FPBits(__llvm_libc::log10f(-1.0f)).is_nan());
  EXPECT_FP_EQ(zero, __llvm_libc::log10f(1.0f));
}

TEST(LlvmLibcLog10fTest, TrickyInputs) {
  constexpr int N = 12;
  constexpr uint32_t INPUTS[N] = {
      0x41200000U /*10.0f*/,
      0x42c80000U /*100.0f*/,
      0x447a0000U /*1,000.0f*/,
      0x461c4000U /*10,000.0f*/,
      0x47c35000U /*100,000.0f*/,
      0x49742400U /*1,000,000.0f*/,
      0x4b189680U /*10,000,000.0f*/,
      0x4cbebc20U /*100,000,000.0f*/,
      0x4e6e6b28U /*1,000,000,000.0f*/,
      0x501502f9U /*10,000,000,000.0f*/,
      0x4f134f83U /*2471461632.0f*/,
      0x7956ba5eU /*69683218960000541503257137270226944.0f*/};

  for (int i = 0; i < N; ++i) {
    float x = float(FPBits(INPUTS[i]));
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log10, x,
                                   __llvm_libc::log10f(x), 0.5);
  }
}

TEST(LlvmLibcLog10fTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    errno = 0;
    float result = __llvm_libc::log10f(x);
    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || errno != 0)
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log10, x,
                                   __llvm_libc::log10f(x), 0.5);
  }
}
