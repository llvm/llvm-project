//===-- Exhaustive test for fmodf16 ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/math/fmodf16.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LlvmLibcFmodf16ExhaustiveTest =
    LlvmLibcBinaryOpExhaustiveMathTest<float16, mpfr::Operation::Fmod,
                                       LIBC_NAMESPACE::fmodf16>;

// Range: [0, Inf];
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7c00U;

// Range: [-Inf, 0];
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xfc00U;

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
  test_full_range_all_roundings(NEG_START, NEG_STOP, POS_START, POS_STOP);
}
