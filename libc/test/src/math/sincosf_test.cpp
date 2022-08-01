//===-- Unittests for sincosf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/sincosf.h"
#include "test/src/math/sdcomp26094.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using __llvm_libc::testing::SDCOMP26094_VALUES;
using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcSinCosfTest, SpecialNumbers) {
  errno = 0;
  float sin, cos;

  __llvm_libc::sincosf(aNaN, &sin, &cos);
  EXPECT_FP_EQ(aNaN, cos);
  EXPECT_FP_EQ(aNaN, sin);
  EXPECT_MATH_ERRNO(0);

  __llvm_libc::sincosf(0.0f, &sin, &cos);
  EXPECT_FP_EQ(1.0f, cos);
  EXPECT_FP_EQ(0.0f, sin);
  EXPECT_MATH_ERRNO(0);

  __llvm_libc::sincosf(-0.0f, &sin, &cos);
  EXPECT_FP_EQ(1.0f, cos);
  EXPECT_FP_EQ(-0.0f, sin);
  EXPECT_MATH_ERRNO(0);

  __llvm_libc::sincosf(inf, &sin, &cos);
  EXPECT_FP_EQ(aNaN, cos);
  EXPECT_FP_EQ(aNaN, sin);
  EXPECT_MATH_ERRNO(EDOM);

  __llvm_libc::sincosf(neg_inf, &sin, &cos);
  EXPECT_FP_EQ(aNaN, cos);
  EXPECT_FP_EQ(aNaN, sin);
  EXPECT_MATH_ERRNO(EDOM);
}

TEST(LlvmLibcSinCosfTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits((v)));
    if (isnan(x) || isinf(x))
      continue;

    float sin, cos;
    __llvm_libc::sincosf(x, &sin, &cos);
    ASSERT_MPFR_MATCH(mpfr::Operation::Cos, x, cos, 1.0);
    ASSERT_MPFR_MATCH(mpfr::Operation::Sin, x, sin, 1.0);
  }
}

// For small values, cos(x) is 1 and sin(x) is x.
TEST(LlvmLibcSinCosfTest, SmallValues) {
  uint32_t bits = 0x17800000;
  float x = float(FPBits((bits)));
  float result_cos, result_sin;
  __llvm_libc::sincosf(x, &result_sin, &result_cos);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cos, x, result_cos, 1.0);
  EXPECT_MPFR_MATCH(mpfr::Operation::Sin, x, result_sin, 1.0);
  EXPECT_FP_EQ(1.0f, result_cos);
  EXPECT_FP_EQ(x, result_sin);

  bits = 0x00400000;
  x = float(FPBits((bits)));
  __llvm_libc::sincosf(x, &result_sin, &result_cos);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cos, x, result_cos, 1.0);
  EXPECT_MPFR_MATCH(mpfr::Operation::Sin, x, result_sin, 1.0);
  EXPECT_FP_EQ(1.0f, result_cos);
  EXPECT_FP_EQ(x, result_sin);
}

// SDCOMP-26094: check sinf in the cases for which the range reducer
// returns values furthest beyond its nominal upper bound of pi/4.
TEST(LlvmLibcSinCosfTest, SDCOMP_26094) {
  for (uint32_t v : SDCOMP26094_VALUES) {
    float x = float(FPBits((v)));
    float sin, cos;
    __llvm_libc::sincosf(x, &sin, &cos);
    EXPECT_MPFR_MATCH(mpfr::Operation::Cos, x, cos, 1.0);
    EXPECT_MPFR_MATCH(mpfr::Operation::Sin, x, sin, 1.0);
  }
}
