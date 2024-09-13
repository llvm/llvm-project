//===-- Unittests for asinf
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/asinf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcAsinfTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcAsinfTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asinf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::asinf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(-0.0f, LIBC_NAMESPACE::asinf(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asinf(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::asinf(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcAsinfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (FPBits(v).is_nan() || FPBits(v).is_inf())
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Asin, x,
                                   LIBC_NAMESPACE::asinf(x), 0.5);
  }
}

TEST_F(LlvmLibcAsinfTest, SpecificBitPatterns) {
  constexpr int N = 11;
  constexpr uint32_t INPUTS[N] = {
      0x3f000000, // x = 0.5f
      0x3f3504f3, // x = sqrt(2)/2, FE_DOWNWARD
      0x3f3504f4, // x = sqrt(2)/2, FE_UPWARD
      0x3f5db3d7, // x = sqrt(3)/2, FE_DOWNWARD
      0x3f5db3d8, // x = sqrt(3)/2, FE_UPWARD
      0x3f800000, // x = 1.0f
      0x40000000, // x = 2.0f
      0x3d09bf86, // x = 0x1.137f0cp-5f
      0x3de5fa1e, // x = 0x1.cbf43cp-4f
      0x3f083a1a, // x = 0x1.107434p-1f
      0x3f7741b6, // x = 0x1.ee836cp-1f
  };

  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Asin, x,
                                   LIBC_NAMESPACE::asinf(x), 0.5);
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Asin, -x,
                                   LIBC_NAMESPACE::asinf(-x), 0.5);
  }
}
