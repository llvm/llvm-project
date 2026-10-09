//===-- Exhaustive test for exp2bf16 --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/exp2bf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcExp2bf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// range: [0, inf]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// range: [-0, -inf]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xff80U;

TEST_F(LlvmLibcExp2bf16Test, PositiveRange) {
  for (uint16_t v = POS_START; v <= POS_STOP; v++) {
    bfloat16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x,
                                   LIBC_NAMESPACE::exp2bf16(x), 0.5);
  }
}

TEST_F(LlvmLibcExp2bf16Test, NegativeRange) {
  for (uint16_t v = NEG_START; v <= NEG_STOP; v++) {
    bfloat16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x,
                                   LIBC_NAMESPACE::exp2bf16(x), 0.5);
  }
}
