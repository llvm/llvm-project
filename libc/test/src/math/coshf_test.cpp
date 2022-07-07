//===-- Unittests for coshf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/Array.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/coshf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcCoshfTest, SpecialNumbers) {
  errno = 0;

  EXPECT_FP_EQ(aNaN, __llvm_libc::coshf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, __llvm_libc::coshf(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, __llvm_libc::coshf(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, __llvm_libc::coshf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, __llvm_libc::coshf(-0.0f));
  EXPECT_MATH_ERRNO(0);
}

TEST(LlvmLibcCoshfTest, Overflow) {
  errno = 0;
  EXPECT_FP_EQ(inf, __llvm_libc::coshf(float(FPBits(0x7f7fffffU))));
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ(inf, __llvm_libc::coshf(float(FPBits(0x42cffff8U))));
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ(inf, __llvm_libc::coshf(float(FPBits(0x42d00008U))));
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST(LlvmLibcCoshfTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Cosh, x, __llvm_libc::coshf(x), 0.5);
  }
}

TEST(LlvmLibcCoshfTest, SmallValues) {
  float x = float(FPBits(0x17800000U));
  float result = __llvm_libc::coshf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cosh, x, result, 0.5);
  EXPECT_FP_EQ(1.0f, result);

  x = float(FPBits(0x0040000U));
  result = __llvm_libc::coshf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cosh, x, result, 0.5);
  EXPECT_FP_EQ(1.0f, result);
}
