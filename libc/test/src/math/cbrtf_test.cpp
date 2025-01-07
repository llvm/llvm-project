//===-- Unittests for cbrtf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/cbrtf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcCbrtfTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcCbrtfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  const uint32_t STEP = FPBits(inf).uintval() / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cbrt, x,
                                   LIBC_NAMESPACE::cbrtf(x), 0.5);
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cbrt, -x,
                                   LIBC_NAMESPACE::cbrtf(-x), 0.5);
  }
}

TEST_F(LlvmLibcCbrtfTest, SpecialValues) {
  constexpr float INPUTS[] = {
      0x1.60451p2f, 0x1.31304cp1f, 0x1.d17cp2f, 0x1.bp-143f, 0x1.338cp2f,
  };
  for (float v : INPUTS) {
    float x = FPBits(v).get_val();
    mpfr::ForceRoundingMode r(mpfr::RoundingMode::Upward);
    EXPECT_MPFR_MATCH(mpfr::Operation::Cbrt, x, LIBC_NAMESPACE::cbrtf(x), 0.5,
                      mpfr::RoundingMode::Upward);
  }
}
