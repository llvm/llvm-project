//===-- Unittests for pow -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "src/math/pow.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcPowTest = LIBC_NAMESPACE::testing::FPTest<double>;
using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

TEST_F(LlvmLibcPowTest, SpecialNumbers) {
  constexpr double NEG_ODD_INTEGER = -3.0;
  constexpr double NEG_EVEN_INTEGER = -6.0;
  constexpr double NEG_NON_INTEGER = -1.1;
  constexpr double POS_ODD_INTEGER = 5.0;
  constexpr double POS_EVEN_INTEGER = 8.0;
  constexpr double POS_NON_INTEGER = 1.1;
  constexpr double ONE_HALF = 0.5;

  for (int i = 0; i < N_ROUNDING_MODES; ++i) {
    ForceRoundingMode __r(ROUNDING_MODES[i]);
    if (!__r.success)
      continue;

    // pow( 0.0, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::pow(zero, NEG_ODD_INTEGER),
                                FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::pow(zero, NEG_EVEN_INTEGER), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::pow(zero, NEG_NON_INTEGER),
                                FE_DIVBYZERO);
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(zero, POS_ODD_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(zero, POS_EVEN_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(zero, POS_NON_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(zero, ONE_HALF));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(zero, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(zero, neg_zero));
    EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::pow(zero, inf));
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::pow(zero, neg_inf),
                                FE_DIVBYZERO);
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(zero, aNaN));

    // pow( -0.0, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(
        neg_inf, LIBC_NAMESPACE::pow(neg_zero, NEG_ODD_INTEGER), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::pow(neg_zero, NEG_EVEN_INTEGER), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::pow(neg_zero, NEG_NON_INTEGER), FE_DIVBYZERO);
    EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::pow(neg_zero, POS_ODD_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(neg_zero, POS_EVEN_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(neg_zero, POS_NON_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(neg_zero, ONE_HALF));
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
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, NEG_ODD_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, NEG_EVEN_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, NEG_NON_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, POS_ODD_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, POS_EVEN_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, POS_NON_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, inf));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, neg_inf));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(1.0, aNaN));

    // pow( 1.0, exponent )
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(-1.0, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(-1.0, neg_zero));
    EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::pow(-1.0, 1.0));
    EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::pow(-1.0, -1.0));
    EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::pow(-1.0, NEG_ODD_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(-1.0, NEG_EVEN_INTEGER));
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::pow(-1.0, NEG_NON_INTEGER),
                                    FE_INVALID);
    EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::pow(-1.0, POS_ODD_INTEGER));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(-1.0, POS_EVEN_INTEGER));
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::pow(-1.0, POS_NON_INTEGER),
                                    FE_INVALID);
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(-1.0, inf));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(-1.0, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(-1.0, aNaN));

    // pow( inf, exponent )
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(inf, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(inf, neg_zero));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(inf, 1.0));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(inf, -1.0));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(inf, NEG_ODD_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(inf, NEG_EVEN_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(inf, NEG_NON_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(inf, POS_ODD_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(inf, POS_EVEN_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(inf, POS_NON_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(inf, ONE_HALF));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(inf, inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(inf, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(inf, aNaN));

    // pow( -inf, exponent )
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(neg_inf, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(neg_inf, neg_zero));
    EXPECT_FP_EQ(neg_inf, LIBC_NAMESPACE::pow(neg_inf, 1.0));
    EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::pow(neg_inf, -1.0));
    EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::pow(neg_inf, NEG_ODD_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(neg_inf, NEG_EVEN_INTEGER));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(neg_inf, NEG_NON_INTEGER));
    EXPECT_FP_EQ(neg_inf, LIBC_NAMESPACE::pow(neg_inf, POS_ODD_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(neg_inf, POS_EVEN_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(neg_inf, POS_NON_INTEGER));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(neg_inf, ONE_HALF));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(neg_inf, inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::pow(neg_inf, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(neg_inf, aNaN));

    // pow ( aNaN, exponent )
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(aNaN, zero));
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(aNaN, neg_zero));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, 1.0));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, -1.0));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, NEG_ODD_INTEGER));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, NEG_EVEN_INTEGER));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, NEG_NON_INTEGER));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, POS_ODD_INTEGER));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, POS_EVEN_INTEGER));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(aNaN, POS_NON_INTEGER));
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

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcPowTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(-min_denormal, 0.5));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(2.0, min_denormal));
}

TEST_F(LlvmLibcPowTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::pow(-min_denormal, 0.5));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(2.0, min_denormal));
}

TEST_F(LlvmLibcPowTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::pow(-min_denormal, 0.5));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(2.0, min_denormal));
}

#endif
