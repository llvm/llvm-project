//===-- Exhaustive test for rsqrtf ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/rsqrtf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcRsqrtfTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Range: [0, Inf]
static constexpr uint32_t POS_START = 0x00000000u;
static constexpr uint32_t POS_STOP = 0x7F800000u;

// Range: [-Inf, 0)
// rsqrt(-0.0) is -inf, not the same for mpfr.
static constexpr uint32_t NEG_START = 0x80000001u;
static constexpr uint32_t NEG_STOP = 0xFF800000u;

TEST_F(LlvmLibcRsqrtfTest, PositiveRange) {
  for (uint32_t v = POS_START; v <= POS_STOP; ++v) {
    float x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Rsqrt, x,
                                   LIBC_NAMESPACE::rsqrtf(x), 0.5);
  }
}

TEST_F(LlvmLibcRsqrtfTest, NegativeRange) {
  for (uint32_t v = NEG_START; v <= NEG_STOP; ++v) {
    float x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Rsqrt, x,
                                   LIBC_NAMESPACE::rsqrtf(x), 0.5);
  }
}
