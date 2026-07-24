//===-- Unittests for lgammabf16 ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/lgammabf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcLgammabf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Subnormal positive range
static constexpr uint16_t SUBNORM_POS_START = 0x0001U;
static constexpr uint16_t SUBNORM_POS_STOP = 0x007FU;

// Normal positive range
static constexpr uint16_t NORMAL_POS_START = 0x0080U;
static constexpr uint16_t NORMAL_POS_STOP = 0x7F7FU;

// Subnormal negative range
static constexpr uint16_t SUBNORM_NEG_START = 0x8001U;
static constexpr uint16_t SUBNORM_NEG_STOP = 0x807FU;

// Normal negative range
static constexpr uint16_t NORMAL_NEG_START = 0x8080U;
static constexpr uint16_t NORMAL_NEG_STOP = 0xFF7FU;

TEST_F(LlvmLibcLgammabf16Test, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::lgammabf16(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::lgammabf16(zero));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::lgammabf16(neg_zero));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::lgammabf16(inf));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::lgammabf16(neg_inf));
  // lgamma(1) = lgamma(2) = 0
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::lgammabf16(bfloat16(1.0f)));
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::lgammabf16(bfloat16(2.0f)));
  // Negative integers are poles -> +inf
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::lgammabf16(bfloat16(-1.0f)));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::lgammabf16(bfloat16(-2.0f)));
}

TEST_F(LlvmLibcLgammabf16Test, SubnormalPositiveRange) {
  for (uint16_t v = SUBNORM_POS_START; v <= SUBNORM_POS_STOP; ++v) {
    bfloat16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Lgamma, x,
                                   LIBC_NAMESPACE::lgammabf16(x), 0.5);
  }
}

TEST_F(LlvmLibcLgammabf16Test, NormalPositiveRange) {
  for (uint16_t v = NORMAL_POS_START; v <= NORMAL_POS_STOP; ++v) {
    bfloat16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Lgamma, x,
                                   LIBC_NAMESPACE::lgammabf16(x), 0.5);
  }
}

TEST_F(LlvmLibcLgammabf16Test, SubnormalNegativeRange) {
  for (uint16_t v = SUBNORM_NEG_START; v <= SUBNORM_NEG_STOP; ++v) {
    bfloat16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Lgamma, x,
                                   LIBC_NAMESPACE::lgammabf16(x), 0.5);
  }
}

TEST_F(LlvmLibcLgammabf16Test, NormalNegativeRange) {
  for (uint16_t v = NORMAL_NEG_START; v <= NORMAL_NEG_STOP; ++v) {
    bfloat16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Lgamma, x,
                                   LIBC_NAMESPACE::lgammabf16(x), 0.5);
  }
}
