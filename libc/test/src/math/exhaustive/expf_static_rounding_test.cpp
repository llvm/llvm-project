//===-- Exhaustive test for expf ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test_static_rounding.h"
// TODO: debug this? Why do we even need this at all?
#undef LIBC_ERRNO_MODE
#include "shared/static_rounding_math.h"
#include "src/__support/math/expf.h"

using LlvmLibcStaticallyRoundedExpfExhaustiveTest =
    LlvmLibcStaticallyRoundedUnaryOpExhaustiveMathTest<
        float, LIBC_NAMESPACE::math::expf,
        LIBC_NAMESPACE::shared::math::static_rounding::expf>;

// Range: [0, Inf];
static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x7f80'0000U;

TEST_F(LlvmLibcStaticallyRoundedExpfExhaustiveTest, PostiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}

// Range: [-Inf, 0];
static constexpr uint32_t NEG_START = 0xb000'0000U;
static constexpr uint32_t NEG_STOP = 0xff80'0000U;

TEST_F(LlvmLibcStaticallyRoundedExpfExhaustiveTest, NegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP);
}
