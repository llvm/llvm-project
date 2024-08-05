//===-- Unittests for pow -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/pow.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcPowTest = LIBC_NAMESPACE::testing::FPTest<double>;
using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

TEST_F(LlvmLibcPowTest, SpecialNumbers) {
  constexpr double neg_odd_integer = -3.0;
  constexpr double neg_even_integer = -6.0;
  constexpr double neg_non_integer = -1.1;
  constexpr double pos_odd_integer = 5.0;
  constexpr double pos_even_integer = 8.0;
  constexpr double pos_non_integer = 1.1;

  for (int i = 0; i < N_ROUNDING_MODES; ++i) {
    ForceRoundingMode __r(ROUNDING_MODES[i]);
    if (!__r.success)
      continue;

    // pow( 0.0, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::pow(zero, neg_odd_integer),
                                FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::pow(zero, neg_even_integer), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::pow(zero, neg_non_integer),
                                FE_DIVBYZERO);
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(zero, pos_odd_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(zero, pos_even_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(zero, pos_non_integer));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(zero, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(zero, neg_zero));
    EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::pow(zero, inf));
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::pow(zero, neg_inf),
                                FE_DIVBYZERO);
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(zero, aNaN));

    // pow( -0.0, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(
        neg_inf, LIBC_NAMESPACE::pow(neg_zero, neg_odd_integer), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::pow(neg_zero, neg_even_integer), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::pow(neg_zero, neg_non_integer), FE_DIVBYZERO);
    EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::pow(neg_zero, pos_odd_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(neg_zero, pos_even_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(neg_zero, pos_non_integer));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(neg_zero, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(neg_zero, neg_zero));
    EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::pow(neg_zero, inf));
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::pow(neg_zero, neg_inf),
                                FE_DIVBYZERO);
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(neg_zero, aNaN));

    // pow( 1.0, exponent )
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, neg_zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, 1.0));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, -1.0));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, neg_odd_integer));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, neg_even_integer));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, neg_non_integer));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, pos_odd_integer));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, pos_even_integer));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, pos_non_integer));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, inf));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, neg_inf));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, aNaN));

    // pow( 1.0, exponent )
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(-1.0, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(-1.0, neg_zero));
    EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::pow(-1.0, 1.0));
    EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::pow(-1.0, -1.0));
    EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::pow(-1.0, neg_odd_integer));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(-1.0, neg_even_integer));
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::pow(-1.0, neg_non_integer),
                                    FE_INVALID);
    EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::pow(-1.0, pos_odd_integer));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(-1.0, pos_even_integer));
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::pow(-1.0, pos_non_integer),
                                    FE_INVALID);
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(-1.0, inf));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(-1.0, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(-1.0, aNaN));

    // pow( inf, exponent )
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(inf, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(inf, neg_zero));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(inf, 1.0));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(inf, -1.0));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(inf, neg_odd_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(inf, neg_even_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(inf, neg_non_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(inf, pos_odd_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(inf, pos_even_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(inf, pos_non_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(inf, inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(inf, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(inf, aNaN));

    // pow( -inf, exponent )
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(neg_inf, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(neg_inf, neg_zero));
    EXPECT_FP_EQ(neg_inf, LIBC_NAMESPACE::pow(neg_inf, 1.0));
    EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::pow(neg_inf, -1.0));
    EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::pow(neg_inf, neg_odd_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(neg_inf, neg_even_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(neg_inf, neg_non_integer));
    EXPECT_FP_EQ(neg_inf, LIBC_NAMESPACE::pow(neg_inf, pos_odd_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(neg_inf, pos_even_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(neg_inf, pos_non_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(neg_inf, inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(neg_inf, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(neg_inf, aNaN));

    // pow ( aNaN, exponent )
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(aNaN, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(aNaN, neg_zero));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, 1.0));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, -1.0));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, neg_odd_integer));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, neg_even_integer));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, neg_non_integer));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, pos_odd_integer));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, pos_even_integer));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, pos_non_integer));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, aNaN));

    // pow ( base, inf )
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(0.1, inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(-0.1, inf));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(1.1, inf));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(-1.1, inf));

    // pow ( base, -inf )
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(0.1, neg_inf));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(-0.1, neg_inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(1.1, neg_inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(-1.1, neg_inf));

    // Exact powers of 2:
    // TODO: Enable these tests when we use exp2.
    // EXPECT_FP_EQ(0x1.0p15, LIBC_NAMESPACE::pow(2.0, 15.0));
    // EXPECT_FP_EQ(0x1.0p126, LIBC_NAMESPACE::pow(2.0, 126.0));
    // EXPECT_FP_EQ(0x1.0p-45, LIBC_NAMESPACE::pow(2.0, -45.0));
    // EXPECT_FP_EQ(0x1.0p-126, LIBC_NAMESPACE::pow(2.0, -126.0));
    // EXPECT_FP_EQ(0x1.0p-149, LIBC_NAMESPACE::pow(2.0, -149.0));

    // Exact powers of 10:
    // TODO: Enable these tests when we use exp10
    // EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(10.0, 0.0));
    // EXPECT_FP_EQ(10.0, LIBC_NAMESPACE::pow(10.0, 1.0));
    // EXPECT_FP_EQ(100.0, LIBC_NAMESPACE::pow(10.0, 2.0));
    // EXPECT_FP_EQ(1000.0, LIBC_NAMESPACE::pow(10.0, 3.0));
    // EXPECT_FP_EQ(10000.0, LIBC_NAMESPACE::pow(10.0, 4.0));
    // EXPECT_FP_EQ(100000.0, LIBC_NAMESPACE::pow(10.0, 5.0));
    // EXPECT_FP_EQ(1000000.0, LIBC_NAMESPACE::pow(10.0, 6.0));
    // EXPECT_FP_EQ(10000000.0, LIBC_NAMESPACE::pow(10.0, 7.0));
    // EXPECT_FP_EQ(100000000.0, LIBC_NAMESPACE::pow(10.0, 8.0));
    // EXPECT_FP_EQ(1000000000.0, LIBC_NAMESPACE::pow(10.0, 9.0));
    // EXPECT_FP_EQ(10000000000.0, LIBC_NAMESPACE::pow(10.0, 10.0));

    // Overflow / Underflow:
    if (ROUNDING_MODES[i] != RoundingMode::Downward &&
        ROUNDING_MODES[i] != RoundingMode::TowardZero) {
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::pow(3.1, 2001.0),
                                  FE_OVERFLOW);
    }
    if (ROUNDING_MODES[i] != RoundingMode::Upward) {
      EXPECT_FP_EQ_WITH_EXCEPTION(0.0, LIBC_NAMESPACE::pow(3.1, -2001.0),
                                  FE_UNDERFLOW);
    }
  }
}
