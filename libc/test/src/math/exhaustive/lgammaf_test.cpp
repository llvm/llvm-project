//===-- Exhaustive test for lgammaf ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/math/lgammaf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcLgammafTest = LIBC_NAMESPACE::testing::FPTest<float>;

using LlvmLibcLgammafExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathTest<float, mpfr::Operation::Lgamma,
                                      LIBC_NAMESPACE::lgammaf>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Positive range: [0, +Inf)
static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x7f80'0000U;

// Negative range: [-0, -Inf)
static constexpr uint32_t NEG_START = 0x8000'0000U;
static constexpr uint32_t NEG_STOP = 0xff80'0000U;

TEST_F(LlvmLibcLgammafExhaustiveTest, PositiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}

TEST_F(LlvmLibcLgammafExhaustiveTest, NegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP);
}
