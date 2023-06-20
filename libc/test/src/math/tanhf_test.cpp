//===-- Unittests for tanhf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/tanhf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcTanhfTest, SpecialNumbers) {
  libc_errno = 0;

  EXPECT_FP_EQ(aNaN, __llvm_libc::tanhf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, __llvm_libc::tanhf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-0.0f, __llvm_libc::tanhf(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, __llvm_libc::tanhf(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-1.0f, __llvm_libc::tanhf(neg_inf));
  EXPECT_MATH_ERRNO(0);
}

TEST(LlvmLibcTanhfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'001;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Tanh, x,
                                   __llvm_libc::tanhf(x), 0.5);
  }
}

TEST(LlvmLibcTanhfTest, ExceptionalValues) {
  constexpr int N = 4;
  constexpr uint32_t INPUTS[N] = {
      0x0040'0000,
      0x1780'0000,
      0x3a12'85ff,
      0x4058'e0a3,
  };

  for (int i = 0; i < N; ++i) {
    float x = float(FPBits(INPUTS[i]));
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Tanh, x,
                                   __llvm_libc::tanhf(x), 0.5);
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Tanh, -x,
                                   __llvm_libc::tanhf(-x), 0.5);
  }
}
