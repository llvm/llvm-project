//===-- Unittests for powf16 ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "src/math/powf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcPowF16Test = LIBC_NAMESPACE::testing::FPTest<float16>;
using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

TEST_F(LlvmLibcPowF16Test, SpecialNumbers) {
  constexpr float16 NEG_ODD_INTEGER = -3.0f16;
  constexpr float16 NEG_EVEN_INTEGER = -6.0f16;
  constexpr float16 NEG_NON_INTEGER = -1.5f16;
  constexpr float16 POS_ODD_INTEGER = 5.0f16;
  constexpr float16 POS_EVEN_INTEGER = 8.0f16;
  constexpr float16 POS_NON_INTEGER = 1.5f16;
  constexpr float16 ONE_HALF = 0.5f16;

  for (int i = 0; i < N_ROUNDING_MODES; ++i) {

    ForceRoundingMode __r(ROUNDING_MODES[i]);
    if (!__r.success)
      continue;

    // powf16( sNaN, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(sNaN, sNaN),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        aNaN, LIBC_NAMESPACE::powf16(sNaN, NEG_ODD_INTEGER), FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        aNaN, LIBC_NAMESPACE::powf16(sNaN, NEG_EVEN_INTEGER), FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        aNaN, LIBC_NAMESPACE::powf16(sNaN, POS_ODD_INTEGER), FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        aNaN, LIBC_NAMESPACE::powf16(sNaN, POS_EVEN_INTEGER), FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(sNaN, ONE_HALF),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(sNaN, zero),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(sNaN, neg_zero),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(sNaN, inf),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(sNaN, neg_inf),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(sNaN, aNaN),
                                FE_INVALID);

    // powf16( 0.0, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(zero, sNaN),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::powf16(zero, NEG_ODD_INTEGER), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::powf16(zero, NEG_EVEN_INTEGER), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::powf16(zero, NEG_NON_INTEGER), FE_DIVBYZERO);
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(zero, POS_ODD_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(zero, POS_EVEN_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(zero, POS_NON_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(zero, ONE_HALF));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(zero, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(zero, neg_zero));
    EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::powf16(zero, inf));
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::powf16(zero, neg_inf),
                                FE_DIVBYZERO);
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(zero, aNaN));

    // powf16( -0.0, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(neg_zero, sNaN),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        neg_inf, LIBC_NAMESPACE::powf16(neg_zero, NEG_ODD_INTEGER),
        FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::powf16(neg_zero, NEG_EVEN_INTEGER), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::powf16(neg_zero, NEG_NON_INTEGER), FE_DIVBYZERO);
    EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::powf16(neg_zero, POS_ODD_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(neg_zero, POS_EVEN_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(neg_zero, POS_NON_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(neg_zero, ONE_HALF));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(neg_zero, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(neg_zero, neg_zero));
    EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::powf16(neg_zero, inf));
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::powf16(neg_zero, neg_inf),
                                FE_DIVBYZERO);
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(neg_zero, aNaN));

    // powf16( 1.0, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(1.0, sNaN),
                                FE_INVALID);
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, neg_zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, 1.0));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, -1.0));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, NEG_ODD_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, NEG_EVEN_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, NEG_NON_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, POS_ODD_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, POS_EVEN_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, POS_NON_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, inf));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, neg_inf));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(1.0, aNaN));

    // powf16( -1.0, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(-1.0, sNaN),
                                FE_INVALID);
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(-1.0, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(-1.0, neg_zero));
    EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::powf16(-1.0, 1.0));
    EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::powf16(-1.0, -1.0));
    EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::powf16(-1.0, NEG_ODD_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(-1.0, NEG_EVEN_INTEGER));
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(
        LIBC_NAMESPACE::powf16(-1.0, NEG_NON_INTEGER), FE_INVALID);
    EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::powf16(-1.0, POS_ODD_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(-1.0, POS_EVEN_INTEGER));
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(
        LIBC_NAMESPACE::powf16(-1.0, POS_NON_INTEGER), FE_INVALID);
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(-1.0, inf));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(-1.0, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(-1.0, aNaN));

    // powf16( inf, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(inf, sNaN),
                                FE_INVALID);
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(inf, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(inf, neg_zero));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(inf, 1.0));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(inf, -1.0));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(inf, NEG_ODD_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(inf, NEG_EVEN_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(inf, NEG_NON_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(inf, POS_ODD_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(inf, POS_EVEN_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(inf, POS_NON_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(inf, ONE_HALF));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(inf, inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(inf, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(inf, aNaN));

    // powf16( -inf, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(neg_inf, sNaN),
                                FE_INVALID);
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(neg_inf, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(neg_inf, neg_zero));
    EXPECT_FP_EQ(neg_inf, LIBC_NAMESPACE::powf16(neg_inf, 1.0));
    EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::powf16(neg_inf, -1.0));
    EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::powf16(neg_inf, NEG_ODD_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(neg_inf, NEG_EVEN_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(neg_inf, NEG_NON_INTEGER));
    EXPECT_FP_EQ(neg_inf, LIBC_NAMESPACE::powf16(neg_inf, POS_ODD_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(neg_inf, POS_EVEN_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(neg_inf, POS_NON_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(neg_inf, ONE_HALF));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(neg_inf, inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(neg_inf, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(neg_inf, aNaN));

    // powf16 ( aNaN, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::powf16(aNaN, sNaN),
                                FE_INVALID);
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(aNaN, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(aNaN, neg_zero));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(aNaN, 1.0));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(aNaN, -1.0));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(aNaN, NEG_ODD_INTEGER));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(aNaN, NEG_EVEN_INTEGER));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(aNaN, NEG_NON_INTEGER));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(aNaN, POS_ODD_INTEGER));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(aNaN, POS_EVEN_INTEGER));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(aNaN, POS_NON_INTEGER));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(aNaN, inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(aNaN, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf16(aNaN, aNaN));

    // powf16 ( base, inf )
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(0.1f16, inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(-0.1f16, inf));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(1.1f16, inf));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(-1.1f16, inf));

    // powf16 ( base, -inf )
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(0.1f16, neg_inf));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf16(-0.1f16, neg_inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(1.1f16, neg_inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf16(-1.1f16, neg_inf));

    // Exact powers of 2:
    // TODO: Enable these tests when we use exp2.
    // EXPECT_FP_EQ(0x1.0p15, LIBC_NAMESPACE::powf16(2.0, 15.0));
    // EXPECT_FP_EQ(0x1.0p126, LIBC_NAMESPACE::powf16(2.0, 126.0));
    // EXPECT_FP_EQ(0x1.0p-45, LIBC_NAMESPACE::powf16(2.0, -45.0));
    // EXPECT_FP_EQ(0x1.0p-126, LIBC_NAMESPACE::powf16(2.0, -126.0));
    // EXPECT_FP_EQ(0x1.0p-149, LIBC_NAMESPACE::powf16(2.0, -149.0));

    // Exact powers of 10:
    // TODO: Enable these tests when we use exp10
    // EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::powf16(10.0, 0.0));
    // EXPECT_FP_EQ(10.0, LIBC_NAMESPACE::powf16(10.0, 1.0));
    // EXPECT_FP_EQ(100.0, LIBC_NAMESPACE::powf16(10.0, 2.0));
    // EXPECT_FP_EQ(1000.0, LIBC_NAMESPACE::powf16(10.0, 3.0));
    // EXPECT_FP_EQ(10000.0, LIBC_NAMESPACE::powf16(10.0, 4.0));
    // EXPECT_FP_EQ(100000.0, LIBC_NAMESPACE::powf16(10.0, 5.0));
    // EXPECT_FP_EQ(1000000.0, LIBC_NAMESPACE::powf16(10.0, 6.0));
    // EXPECT_FP_EQ(10000000.0, LIBC_NAMESPACE::powf16(10.0, 7.0));
    // EXPECT_FP_EQ(100000000.0, LIBC_NAMESPACE::powf16(10.0, 8.0));
    // EXPECT_FP_EQ(1000000000.0, LIBC_NAMESPACE::powf16(10.0, 9.0));
    // EXPECT_FP_EQ(10000000000.0, LIBC_NAMESPACE::powf16(10.0, 10.0));

    // Overflow / Underflow:
    if (ROUNDING_MODES[i] != RoundingMode::Downward &&
        ROUNDING_MODES[i] != RoundingMode::TowardZero) {
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::powf16(3.1f16, 21.0),
                                  FE_OVERFLOW);
    }
    if (ROUNDING_MODES[i] != RoundingMode::Upward) {
      EXPECT_FP_EQ_WITH_EXCEPTION(0.0, LIBC_NAMESPACE::powf16(3.1f16, -21.0),
                                  FE_UNDERFLOW);
    }
  }
}
