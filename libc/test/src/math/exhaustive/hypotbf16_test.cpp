//===-- Exhaustive test for hypotbf16 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/hypotbf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcHypotBf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Normal range: [+0, +int]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// Normal range: [-0, -int]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xff80U;

TEST_F(LlvmLibcHypotBf16Test, PositiveRange) {
  for (uint16_t v1 = POS_START; v1 <= POS_STOP; v1++) {
    for (uint16_t v2 = v1; v2 <= POS_STOP; v2++) {

      bfloat16 x = FPBits(v1).get_val();
      bfloat16 y = FPBits(v2).get_val();
      mpfr::BinaryInput<bfloat16> input{x, y};

      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Hypot, input,
                                     LIBC_NAMESPACE::hypotbf16(x, y), 0.5);
    }
  }
}

TEST_F(LlvmLibcHypotBf16Test, NegativeRange) {
  for (uint16_t v1 = NEG_START; v1 <= NEG_STOP; v1++) {
    for (uint16_t v2 = v1; v2 <= NEG_STOP; v2++) {

      bfloat16 x = FPBits(v1).get_val();
      bfloat16 y = FPBits(v2).get_val();
      mpfr::BinaryInput<bfloat16> input{x, y};

      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Hypot, input,
                                     LIBC_NAMESPACE::hypotbf16(x, y), 0.5);
    }
  }
}

TEST_F(LlvmLibcHypotBf16Test, SpecialNumbers) {
  constexpr bfloat16 VALUES[] = {zero,    neg_zero,   inf,
                                 neg_inf, min_normal, max_normal};
  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = i; j < 6; ++j) {
      bfloat16 x = VALUES[i];
      bfloat16 y = VALUES[j];
      mpfr::BinaryInput<bfloat16> input{x, y};

      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Hypot, input,
                                     LIBC_NAMESPACE::hypotbf16(x, y), 0.5);
    }
  }
}
