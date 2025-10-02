//===-- Exhaustive test for fmodbf16 --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/fmodbf16.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LlvmLibcFmodf16ExhaustiveTest =
    LlvmLibcBinaryOpExhaustiveMathTest<bfloat16, mpfr::Operation::Fmod,
                                       LIBC_NAMESPACE::fmodbf16>;

// range: [0, inf]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// range: [-0, -inf]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xff80U;

TEST_F(LlvmLibcFmodf16ExhaustiveTest, PostivePositiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP, POS_START, POS_STOP);
}

TEST_F(LlvmLibcFmodf16ExhaustiveTest, PostiveNegativeRange) {
  test_full_range_all_roundings(POS_START, POS_STOP, NEG_START, NEG_STOP);
}

TEST_F(LlvmLibcFmodf16ExhaustiveTest, NegativePositiveRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP, POS_START, POS_STOP);
}

TEST_F(LlvmLibcFmodf16ExhaustiveTest, NegativeNegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP, NEG_START, NEG_STOP);
}
