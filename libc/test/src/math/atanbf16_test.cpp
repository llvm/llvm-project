//===-- Unittests for atanbf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/atanbf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcAtanBf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Normal range: [+0, +int]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// Normal range: [-0, -int]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xff80U;

TEST_F(LlvmLibcAtanBf16Test, NormalPositiveRange) {
  for (uint16_t v1 = POS_START; v1 <= POS_STOP; v1++) {

    bfloat16 x = FPBits(v1).get_val();

    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan, x,
                                   LIBC_NAMESPACE::atanbf16(x), 0.5);
  }
}

TEST_F(LlvmLibcAtanBf16Test, NormalNegativeRange) {
  for (uint16_t v1 = NEG_START; v1 <= NEG_STOP; v1++) {

    bfloat16 x = FPBits(v1).get_val();

    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan, x,
                                   LIBC_NAMESPACE::atanbf16(x), 0.5);
  }
}

TEST_F(LlvmLibcAtanBf16Test, SpecialNumbers) {
  constexpr bfloat16 VALUES[] = {zero,    neg_zero,   inf,
                                 neg_inf, min_normal, max_normal};
  for (size_t i = 0; i < 6; ++i) {
    bfloat16 x = VALUES[i];

    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan, x,
                                   LIBC_NAMESPACE::atanbf16(x), 0.5);
  }
}
