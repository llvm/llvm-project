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

using LlvmLibcLog10fTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcLog10fTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::log10f(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::log10f(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log10f(neg_inf), FE_INVALID);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log10f(0.0f),
                              FE_DIVBYZERO);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log10f(-0.0f),
                              FE_DIVBYZERO);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log10f(-1.0f), FE_INVALID);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::log10f(1.0f));
}

TEST_F(LlvmLibcLog10fTest, TrickyInputs) {
  constexpr int N = 21;
  constexpr uint32_t INPUTS[N] = {
      0x3f800000U /*1.0f*/,
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
      0x7956ba5eU /*69683218960000541503257137270226944.0f*/,
      0x08ae'a356U /*0x1.5d46acp-110f*/,
      0x1c7d'a337U /*0x1.fb466ep-71f*/,
      0x69c8'c583U /*0x1.918b06p+84f*/,
      0x0efe'ee7aU /*0x1.fddcf4p-98f*/,
      0x3f5f'de1bU /*0x1.bfbc36p-1f*/,
      0x3f80'70d8U /*0x1.00e1bp0f*/,
      0x120b'93dcU /*0x1.1727b8p-91f*/,
      0x13ae'78d3U /*0x1.5cf1a6p-88f*/,
  };

  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log10, x,
                                   LIBC_NAMESPACE::log10f(x), 0.5);
  }
}

TEST_F(LlvmLibcLog10fTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log10, x,
                                   LIBC_NAMESPACE::log10f(x), 0.5);
  }
}
