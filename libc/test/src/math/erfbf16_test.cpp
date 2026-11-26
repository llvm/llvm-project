//===-- Exhaustive test for erfbf16 ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/erfbf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcErfbf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Range: [+0, +Inf].
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// Range: [-0, -Inf].
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xff80U;

TEST_F(LlvmLibcErfbf16Test, PositiveRange) {
  for (uint32_t bits = POS_START; bits <= POS_STOP; ++bits) {
    bfloat16 x = FPBits(static_cast<uint16_t>(bits)).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Erf, x,
                                   LIBC_NAMESPACE::erfbf16(x), 0.5);
  }
}

TEST_F(LlvmLibcErfbf16Test, NegativeRange) {
  for (uint32_t bits = NEG_START; bits <= NEG_STOP; ++bits) {
    bfloat16 x = FPBits(static_cast<uint16_t>(bits)).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Erf, x,
                                   LIBC_NAMESPACE::erfbf16(x), 0.5);
  }
}
