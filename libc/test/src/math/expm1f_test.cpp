//===-- Unittests for expm1f-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/expm1f.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcExpm1fTest, SpecialNumbers) {
  errno = 0;

  EXPECT_FP_EQ(aNaN, __llvm_libc::expm1f(aNaN));
  EXPECT_EQ(errno, 0);

  EXPECT_FP_EQ(inf, __llvm_libc::expm1f(inf));
  EXPECT_EQ(errno, 0);

  EXPECT_FP_EQ(-1.0f, __llvm_libc::expm1f(neg_inf));
  EXPECT_EQ(errno, 0);

  EXPECT_FP_EQ(0.0f, __llvm_libc::expm1f(0.0f));
  EXPECT_EQ(errno, 0);

  EXPECT_FP_EQ(-0.0f, __llvm_libc::expm1f(-0.0f));
  EXPECT_EQ(errno, 0);
}

TEST(LlvmLibcExpm1fTest, Overflow) {
  errno = 0;
  EXPECT_FP_EQ(inf, __llvm_libc::expm1f(float(FPBits(0x7f7fffffU))));
  EXPECT_EQ(errno, ERANGE);

  errno = 0;
  EXPECT_FP_EQ(inf, __llvm_libc::expm1f(float(FPBits(0x42cffff8U))));
  EXPECT_EQ(errno, ERANGE);

  errno = 0;
  EXPECT_FP_EQ(inf, __llvm_libc::expm1f(float(FPBits(0x42d00008U))));
  EXPECT_EQ(errno, ERANGE);
}

TEST(LlvmLibcExpm1fTest, Underflow) {
  errno = 0;
  EXPECT_FP_EQ(-1.0f, __llvm_libc::expm1f(float(FPBits(0xff7fffffU))));
  EXPECT_EQ(errno, ERANGE);

  errno = 0;
  float x = float(FPBits(0xc2cffff8U));
  EXPECT_FP_EQ(-1.0f, __llvm_libc::expm1f(x));
  EXPECT_EQ(errno, ERANGE);

  errno = 0;
  x = float(FPBits(0xc2d00008U));
  EXPECT_FP_EQ(-1.0f, __llvm_libc::expm1f(x));
  EXPECT_EQ(errno, ERANGE);
}

// Test with inputs which are the borders of underflow/overflow but still
// produce valid results without setting errno.
TEST(LlvmLibcExpm1fTest, Borderline) {
  float x;

  errno = 0;
  x = float(FPBits(0x42affff8U));
  ASSERT_MPFR_MATCH(mpfr::Operation::Expm1, x, __llvm_libc::expm1f(x), 1.0);
  EXPECT_EQ(errno, 0);

  x = float(FPBits(0x42b00008U));
  ASSERT_MPFR_MATCH(mpfr::Operation::Expm1, x, __llvm_libc::expm1f(x), 1.0);
  EXPECT_EQ(errno, 0);

  x = float(FPBits(0xc2affff8U));
  ASSERT_MPFR_MATCH(mpfr::Operation::Expm1, x, __llvm_libc::expm1f(x), 1.0);
  EXPECT_EQ(errno, 0);

  x = float(FPBits(0xc2b00008U));
  ASSERT_MPFR_MATCH(mpfr::Operation::Expm1, x, __llvm_libc::expm1f(x), 1.0);
  EXPECT_EQ(errno, 0);
}

TEST(LlvmLibcExpm1fTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    errno = 0;
    float result = __llvm_libc::expm1f(x);

    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || errno != 0)
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Expm1, x, __llvm_libc::expm1f(x), 2.2);
  }
}
