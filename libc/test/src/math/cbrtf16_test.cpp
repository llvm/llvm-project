//===-- Unittests for cbrtf16 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/cbrtf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcCbrtf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Range: [0, Inf];
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7c00U;

// Range: [-Inf, 0]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xfc00U;

TEST_F(LlvmLibcCbrtf16Test, PositiveRange) {
  for (uint16_t v = POS_START; v <= POS_STOP; ++v) {
    float16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cbrt, x,
                                   LIBC_NAMESPACE::cbrtf16(x), 0.5);
  }
}

TEST_F(LlvmLibcCbrtf16Test, NegativeRange) {
  for (uint16_t v = NEG_START; v <= NEG_STOP; ++v) {
    float16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cbrt, x,
                                   LIBC_NAMESPACE::cbrtf16(x), 0.5);
  }
}

TEST_F(LlvmLibcCbrtf16Test, SpecialValues) {
  constexpr uint16_t INPUTS[] = {
      0x4a00, 0x4500, 0x4e00, 0x4940, 0x0c00,
      0x4248, // ~pi
      0x7bff, // max positive value
      0x3800, // 0.5
      0x3bec, // ~0.99
      0x3c01, // ~1.01
  };
  for (uint16_t v : INPUTS) {
    float16 x = FPBits(v).get_val();
    mpfr::ForceRoundingMode r(mpfr::RoundingMode::Upward);
    EXPECT_MPFR_MATCH(mpfr::Operation::Cbrt, x, LIBC_NAMESPACE::cbrtf16(x), 0.5,
                      mpfr::RoundingMode::Upward);
  }
}
