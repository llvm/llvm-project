//===-- Unittests for cosf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/Array.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/cosf.h"
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

TEST(LlvmLibcCosfTest, SpecialNumbers) {
  errno = 0;

  EXPECT_FP_EQ(aNaN, __llvm_libc::cosf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, __llvm_libc::cosf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, __llvm_libc::cosf(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, __llvm_libc::cosf(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, __llvm_libc::cosf(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST(LlvmLibcCosfTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Cos, x, __llvm_libc::cosf(x), 1.0);
  }
}

// For small values, cos(x) is 1.
TEST(LlvmLibcCosfTest, SmallValues) {
  float x = float(FPBits(0x17800000U));
  float result = __llvm_libc::cosf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cos, x, result, 1.0);
  EXPECT_FP_EQ(1.0f, result);

  x = float(FPBits(0x0040000U));
  result = __llvm_libc::cosf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cos, x, result, 1.0);
  EXPECT_FP_EQ(1.0f, result);
}

// SDCOMP-26094: check cosf in the cases for which the range reducer
// returns values furthest beyond its nominal upper bound of pi/4.
TEST(LlvmLibcCosfTest, SDCOMP_26094) {
  for (uint32_t v : SDCOMP26094_VALUES) {
    float x = float(FPBits(v));
    ASSERT_MPFR_MATCH(mpfr::Operation::Cos, x, __llvm_libc::cosf(x), 1.0);
  }
}
