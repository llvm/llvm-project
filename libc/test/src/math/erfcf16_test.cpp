//===-- Exhaustive test for erfcf16 ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/optimization.h"
#include "src/math/erfcf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcErfcf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Range: [0, Inf];
// 0x0000 is +0.0, 0x7c00 is +Inf.
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7c00U;

// Range: [-0, -Inf];
// 0x8000 is -0.0, 0xfc00 is -Inf.
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xfc00U;

TEST_F(LlvmLibcErfcf16Test, PositiveRange) {
  for (uint16_t v = POS_START; v <= POS_STOP; ++v) {
    float16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Erfc, x,
                                   LIBC_NAMESPACE::erfcf16(x), 0.5);
  }
}

TEST_F(LlvmLibcErfcf16Test, NegativeRange) {
  for (uint16_t v = NEG_START; v <= NEG_STOP; ++v) {
    float16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Erfc, x,
                                   LIBC_NAMESPACE::erfcf16(x), 0.5);
  }
}
