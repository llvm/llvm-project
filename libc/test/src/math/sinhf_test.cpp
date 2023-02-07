//===-- Unittests for sinhf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/sinhf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcSinhfTest, SpecialNumbers) {
  errno = 0;

  EXPECT_FP_EQ(aNaN, __llvm_libc::sinhf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, __llvm_libc::sinhf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-0.0f, __llvm_libc::sinhf(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, __llvm_libc::sinhf(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(neg_inf, __llvm_libc::sinhf(neg_inf));
  EXPECT_MATH_ERRNO(0);
}

TEST(LlvmLibcSinhfTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Sinh, x, __llvm_libc::sinhf(x), 0.5);
  }
}

// For small values, sinh(x) is x.
TEST(LlvmLibcSinhfTest, SmallValues) {
  float x = float(FPBits(uint32_t(0x17800000)));
  float result = __llvm_libc::sinhf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Sinh, x, result, 0.5);
  EXPECT_FP_EQ(x, result);

  x = float(FPBits(uint32_t(0x00400000)));
  result = __llvm_libc::sinhf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Sinh, x, result, 0.5);
  EXPECT_FP_EQ(x, result);
}

TEST(LlvmLibcSinhfTest, Overflow) {
  errno = 0;
  EXPECT_FP_EQ(inf, __llvm_libc::sinhf(float(FPBits(0x7f7fffffU))));
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ(inf, __llvm_libc::sinhf(float(FPBits(0x42cffff8U))));
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ(inf, __llvm_libc::sinhf(float(FPBits(0x42d00008U))));
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST(LlvmLibcSinhfTest, ExceptionalValues) {
  float x = float(FPBits(uint32_t(0x3a12'85ffU)));
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinh, x,
                                 __llvm_libc::sinhf(x), 0.5);

  x = -float(FPBits(uint32_t(0x3a12'85ffU)));
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinh, x,
                                 __llvm_libc::sinhf(x), 0.5);
}
