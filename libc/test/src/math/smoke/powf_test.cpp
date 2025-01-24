//===-- Unittests for powf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/powf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

using LlvmLibcPowfTest = LIBC_NAMESPACE::testing::FPTest<float>;
using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

TEST_F(LlvmLibcPowfTest, SpecialNumbers) {
  constexpr float neg_odd_integer = -3.0f;
  constexpr float neg_even_integer = -6.0f;
  constexpr float neg_non_integer = -1.1f;
  constexpr float pos_odd_integer = 5.0f;
  constexpr float pos_even_integer = 8.0f;
  constexpr float pos_non_integer = 1.3f;
  constexpr float one_half = 0.5f;

  for (int i = 0; i < N_ROUNDING_MODES; ++i) {
    ForceRoundingMode __r(ROUNDING_MODES[i]);
    if (!__r.success)
      continue;

    // pow( 0.0f, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::powf(zero, neg_odd_integer), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::powf(zero, neg_even_integer), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::powf(zero, neg_non_integer), FE_DIVBYZERO);
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(zero, pos_odd_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(zero, pos_even_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(zero, pos_non_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(zero, one_half));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(zero, zero));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(zero, neg_zero));
    EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::powf(zero, inf));
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::powf(zero, neg_inf),
                                FE_DIVBYZERO);
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(zero, aNaN));

    // pow( -0.0f, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(
        neg_inf, LIBC_NAMESPACE::powf(neg_zero, neg_odd_integer), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::powf(neg_zero, neg_even_integer), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, LIBC_NAMESPACE::powf(neg_zero, neg_non_integer), FE_DIVBYZERO);
    EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::powf(neg_zero, pos_odd_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(neg_zero, pos_even_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(neg_zero, pos_non_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(neg_zero, one_half));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(neg_zero, zero));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(neg_zero, neg_zero));
    EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::powf(neg_zero, inf));
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::powf(neg_zero, neg_inf),
                                FE_DIVBYZERO);
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(neg_zero, aNaN));

    // pow( 1.0f, exponent )
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, zero));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, neg_zero));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, 1.0f));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, -1.0f));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, neg_odd_integer));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, neg_even_integer));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, neg_non_integer));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, pos_odd_integer));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, pos_even_integer));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, pos_non_integer));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, inf));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, neg_inf));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(1.0f, aNaN));

    // pow( 1.0f, exponent )
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(-1.0f, zero));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(-1.0f, neg_zero));
    EXPECT_FP_EQ(-1.0f, LIBC_NAMESPACE::powf(-1.0f, 1.0f));
    EXPECT_FP_EQ(-1.0f, LIBC_NAMESPACE::powf(-1.0f, -1.0f));
    EXPECT_FP_EQ(-1.0f, LIBC_NAMESPACE::powf(-1.0f, neg_odd_integer));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(-1.0f, neg_even_integer));
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(
        LIBC_NAMESPACE::powf(-1.0f, neg_non_integer), FE_INVALID);
    EXPECT_FP_EQ(-1.0f, LIBC_NAMESPACE::powf(-1.0f, pos_odd_integer));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(-1.0f, pos_even_integer));
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(
        LIBC_NAMESPACE::powf(-1.0f, pos_non_integer), FE_INVALID);
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(-1.0f, inf));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(-1.0f, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(-1.0f, aNaN));

    // pow( inf, exponent )
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(inf, zero));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(inf, neg_zero));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(inf, 1.0f));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(inf, -1.0f));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(inf, neg_odd_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(inf, neg_even_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(inf, neg_non_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(inf, pos_odd_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(inf, pos_even_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(inf, pos_non_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(inf, one_half));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(inf, inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(inf, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(inf, aNaN));

    // pow( -inf, exponent )
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(neg_inf, zero));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(neg_inf, neg_zero));
    EXPECT_FP_EQ(neg_inf, LIBC_NAMESPACE::powf(neg_inf, 1.0f));
    EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::powf(neg_inf, -1.0f));
    EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::powf(neg_inf, neg_odd_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(neg_inf, neg_even_integer));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(neg_inf, neg_non_integer));
    EXPECT_FP_EQ(neg_inf, LIBC_NAMESPACE::powf(neg_inf, pos_odd_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(neg_inf, pos_even_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(neg_inf, pos_non_integer));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(neg_inf, one_half));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(neg_inf, inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(neg_inf, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(neg_inf, aNaN));

    // pow ( aNaN, exponent )
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(aNaN, zero));
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(aNaN, neg_zero));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(aNaN, 1.0f));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(aNaN, -1.0f));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(aNaN, neg_odd_integer));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(aNaN, neg_even_integer));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(aNaN, neg_non_integer));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(aNaN, pos_odd_integer));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(aNaN, pos_even_integer));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(aNaN, pos_non_integer));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(aNaN, inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(aNaN, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(aNaN, aNaN));

    // pow ( base, inf )
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(0.1f, inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(-0.1f, inf));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(1.1f, inf));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(-1.1f, inf));

    // pow ( base, -inf )
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(0.1f, neg_inf));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::powf(-0.1f, neg_inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(1.1f, neg_inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::powf(-1.1f, neg_inf));

    // Exact powers of 2:
    EXPECT_FP_EQ(0x1.0p15f, LIBC_NAMESPACE::powf(2.0f, 15.0f));
    EXPECT_FP_EQ(0x1.0p126f, LIBC_NAMESPACE::powf(2.0f, 126.0f));
    EXPECT_FP_EQ(0x1.0p-45f, LIBC_NAMESPACE::powf(2.0f, -45.0f));
    EXPECT_FP_EQ(0x1.0p-126f, LIBC_NAMESPACE::powf(2.0f, -126.0f));
    EXPECT_FP_EQ(0x1.0p-149f, LIBC_NAMESPACE::powf(2.0f, -149.0f));

    // Exact powers of 10:
    EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(10.0f, 0.0f));
    EXPECT_FP_EQ(10.0f, LIBC_NAMESPACE::powf(10.0f, 1.0f));
    EXPECT_FP_EQ(100.0f, LIBC_NAMESPACE::powf(10.0f, 2.0f));
    EXPECT_FP_EQ(1000.0f, LIBC_NAMESPACE::powf(10.0f, 3.0f));
    EXPECT_FP_EQ(10000.0f, LIBC_NAMESPACE::powf(10.0f, 4.0f));
    EXPECT_FP_EQ(100000.0f, LIBC_NAMESPACE::powf(10.0f, 5.0f));
    EXPECT_FP_EQ(1000000.0f, LIBC_NAMESPACE::powf(10.0f, 6.0f));
    EXPECT_FP_EQ(10000000.0f, LIBC_NAMESPACE::powf(10.0f, 7.0f));
    EXPECT_FP_EQ(100000000.0f, LIBC_NAMESPACE::powf(10.0f, 8.0f));
    EXPECT_FP_EQ(1000000000.0f, LIBC_NAMESPACE::powf(10.0f, 9.0f));
    EXPECT_FP_EQ(10000000000.0f, LIBC_NAMESPACE::powf(10.0f, 10.0f));

    // Overflow / Underflow:
    if (ROUNDING_MODES[i] != RoundingMode::Downward &&
        ROUNDING_MODES[i] != RoundingMode::TowardZero) {
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::powf(3.1f, 201.0f),
                                  FE_OVERFLOW);
    }
    if (ROUNDING_MODES[i] != RoundingMode::Upward) {
      EXPECT_FP_EQ_WITH_EXCEPTION(0.0f, LIBC_NAMESPACE::powf(3.1f, -201.0f),
                                  FE_UNDERFLOW);
    }
  }

  EXPECT_FP_EQ(-0.0f, LIBC_NAMESPACE::powf(-0.015625f, 25.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::powf(-0.015625f, 26.0f));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcPowfTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::powf(-min_denormal, 0.5f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(2.0f, min_denormal));
}

TEST_F(LlvmLibcPowfTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::powf(-min_denormal, 0.5f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(2.0f, min_denormal));
}

TEST_F(LlvmLibcPowfTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::powf(-min_denormal, 0.5f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::powf(2.0f, min_denormal));
}

#endif
