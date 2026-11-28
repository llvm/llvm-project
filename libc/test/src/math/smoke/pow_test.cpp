//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for pow.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/math/pow.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/RoundingModeUtils.h"
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

    // pow( sNaN, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(sNaN, sNaN),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        aNaN, LIBC_NAMESPACE::pow(sNaN, NEG_ODD_INTEGER), FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        aNaN, LIBC_NAMESPACE::pow(sNaN, NEG_EVEN_INTEGER), FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        aNaN, LIBC_NAMESPACE::pow(sNaN, POS_ODD_INTEGER), FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        aNaN, LIBC_NAMESPACE::pow(sNaN, POS_EVEN_INTEGER), FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(sNaN, ONE_HALF),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(sNaN, zero),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(sNaN, neg_zero),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(sNaN, inf),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(sNaN, neg_inf),
                                FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(sNaN, aNaN),
                                FE_INVALID);

    // pow( 0.0, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(zero, sNaN),
                                FE_INVALID);
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
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(zero, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(zero, aNaN));

    // pow( -0.0, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(neg_zero, sNaN),
                                FE_INVALID);
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
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::pow(neg_zero, neg_inf));
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::pow(neg_zero, aNaN));

    // pow( 1.0, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(1.0, sNaN),
                                FE_INVALID);
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

    // pow( -1.0, exponent )
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(-1.0, sNaN),
                                FE_INVALID);
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
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(inf, sNaN),
                                FE_INVALID);
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
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(neg_inf, sNaN),
                                FE_INVALID);
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
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::pow(aNaN, sNaN),
                                FE_INVALID);
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
    EXPECT_FP_EQ(0x1.0p15, LIBC_NAMESPACE::pow(2.0, 15.0));
    EXPECT_FP_EQ(0x1.0p126, LIBC_NAMESPACE::pow(2.0, 126.0));
    EXPECT_FP_EQ(0x1.0p-45, LIBC_NAMESPACE::pow(2.0, -45.0));
    EXPECT_FP_EQ(0x1.0p-126, LIBC_NAMESPACE::pow(2.0, -126.0));
    EXPECT_FP_EQ(0x1.0p-149, LIBC_NAMESPACE::pow(2.0, -149.0));

    // Exact powers of 10:
    EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::pow(10.0, 0.0));
#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
    if (ROUNDING_MODES[i] == RoundingMode::Nearest) {
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS
      EXPECT_FP_EQ(10.0, LIBC_NAMESPACE::pow(10.0, 1.0));
      EXPECT_FP_EQ(100.0, LIBC_NAMESPACE::pow(10.0, 2.0));
      EXPECT_FP_EQ(1000.0, LIBC_NAMESPACE::pow(10.0, 3.0));
      EXPECT_FP_EQ(10000.0, LIBC_NAMESPACE::pow(10.0, 4.0));
      EXPECT_FP_EQ(100000.0, LIBC_NAMESPACE::pow(10.0, 5.0));
      EXPECT_FP_EQ(1000000.0, LIBC_NAMESPACE::pow(10.0, 6.0));
      EXPECT_FP_EQ(10000000.0, LIBC_NAMESPACE::pow(10.0, 7.0));
      EXPECT_FP_EQ(100000000.0, LIBC_NAMESPACE::pow(10.0, 8.0));
      EXPECT_FP_EQ(1000000000.0, LIBC_NAMESPACE::pow(10.0, 9.0));
      EXPECT_FP_EQ(10000000000.0, LIBC_NAMESPACE::pow(10.0, 10.0));
#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
    }
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

    // Overflow / Underflow:
    double expected_pos_ovf = (ROUNDING_MODES[i] == RoundingMode::Downward ||
                               ROUNDING_MODES[i] == RoundingMode::TowardZero)
                                  ? max_normal
                                  : inf;
    double expected_neg_ovf = (ROUNDING_MODES[i] == RoundingMode::Upward ||
                               ROUNDING_MODES[i] == RoundingMode::TowardZero)
                                  ? neg_max_normal
                                  : neg_inf;
    double expected_pos_udf =
        (ROUNDING_MODES[i] == RoundingMode::Upward) ? min_denormal : zero;
    double expected_neg_udf = (ROUNDING_MODES[i] == RoundingMode::Downward)
                                  ? -min_denormal
                                  : neg_zero;

    // Fast-path extreme-exponent early returns (|y * log2(|x|)| ~ 3268.0, odd
    // integer y = +-2001):
    EXPECT_FP_EQ_WITH_EXCEPTION(expected_pos_ovf,
                                LIBC_NAMESPACE::pow(3.1, 2001.0),
                                FE_OVERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);
    EXPECT_FP_EQ_WITH_EXCEPTION(expected_neg_ovf,
                                LIBC_NAMESPACE::pow(-3.1, 2001.0),
                                FE_OVERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);
    EXPECT_FP_EQ_WITH_EXCEPTION(expected_pos_udf,
                                LIBC_NAMESPACE::pow(3.1, -2001.0),
                                FE_UNDERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);
    EXPECT_FP_EQ_WITH_EXCEPTION(expected_neg_udf,
                                LIBC_NAMESPACE::pow(-3.1, -2001.0),
                                FE_UNDERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
    // Special-input power-of-2^(+-2^n) shortcut (4^512 = 2^1024, 4^-538 =
    // 2^-1076):
    EXPECT_FP_EQ_WITH_EXCEPTION(expected_pos_ovf,
                                LIBC_NAMESPACE::pow(4.0, 512.0),
                                FE_OVERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);
    EXPECT_FP_EQ_WITH_EXCEPTION(expected_pos_udf,
                                LIBC_NAMESPACE::pow(4.0, -538.0),
                                FE_UNDERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);

    // Fast-path shifted-exponent (scale = 2^600) overflow when Ziv's test
    // passes (y * log2(|x|) ~ 1024.55 in (1024, 1025), odd integer y = 667):
    EXPECT_FP_EQ_WITH_EXCEPTION(expected_pos_ovf,
                                LIBC_NAMESPACE::pow(2.9, 667.0),
                                FE_OVERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);
    EXPECT_FP_EQ_WITH_EXCEPTION(expected_neg_ovf,
                                LIBC_NAMESPACE::pow(-2.9, 667.0),
                                FE_OVERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);

    // 128-bit accurate-pass overflow: y * log2(|x|) ~ 1024.66 near an exact
    // 53-bit significand boundary fails Ziv's test in directed rounding (odd
    // integer y = 1119):
    EXPECT_FP_EQ_WITH_EXCEPTION(
        expected_pos_ovf,
        LIBC_NAMESPACE::pow(0x1.e2efc8289032dp+0, 0x1.17cp+10),
        FE_OVERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        expected_neg_ovf,
        LIBC_NAMESPACE::pow(-0x1.e2efc8289032dp+0, 0x1.17cp+10),
        FE_OVERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);

    // Fast-path subnormal (ziv_test_denorm) underflow (y * log2(|x|) ~ -1075.91
    // in (-1076, -1075), odd integer y = 1357):
    EXPECT_FP_EQ_WITH_EXCEPTION(
        expected_pos_udf,
        LIBC_NAMESPACE::pow(0x1.27867770a2afep-1, 0x1.534p+10),
        FE_UNDERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        expected_neg_udf,
        LIBC_NAMESPACE::pow(-0x1.27867770a2afep-1, 0x1.534p+10),
        FE_UNDERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);

    // Accurate-pass (128-bit and 256-bit) subnormal underflow: y * log2(|x|) ~
    // -1075 - 2^-53.9 right below the 2^-1075 halfway boundary fails
    // ziv_test_denorm in FE_TONEAREST (odd integer y = 32845):
    EXPECT_FP_EQ_WITH_EXCEPTION(
        expected_pos_udf,
        LIBC_NAMESPACE::pow(0x1.f483ee2af72acp-1, 0x1.009ap+15),
        FE_UNDERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        expected_neg_udf,
        LIBC_NAMESPACE::pow(-0x1.f483ee2af72acp-1, 0x1.009ap+15),
        FE_UNDERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);
#endif // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  }
}

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
TEST_F(LlvmLibcPowTest, HardToRound) {
  // Tiny |y * log2(x)| < 2^-54 (z_exp <= -55 shortcut in pow_accurate):
  EXPECT_FP_EQ_ALL_ROUNDING(
      1.0, 0x1.0000000000001p+0, 1.0, 1.0,
      LIBC_NAMESPACE::pow(0x1.0000000000001p+0, 0x1.0p-20));
  EXPECT_FP_EQ_ALL_ROUNDING(
      1.0, 1.0, 0x1.fffffffffffffp-1, 0x1.fffffffffffffp-1,
      LIBC_NAMESPACE::pow(0x1.fffffffffffffp-1, 0x1.0p-20));

  // Exact 54-bit halfway midpoint (is_exact_rounding_boundary: x^(3/2) =
  // 0x1.0006c65a241cb8p+5):
  EXPECT_FP_EQ_ALL_ROUNDING(0x1.0006c65a241ccp+5, 0x1.0006c65a241ccp+5,
                            0x1.0006c65a241cbp+5, 0x1.0006c65a241cbp+5,
                            LIBC_NAMESPACE::pow(0x1.428fe0562p+3, 0x1.8p+0));

  // Near-1.0 bases exercising leading-zero normalization in log2_f128 and
  // log2_f256 (x < 1 and x > 1, distance < 2^-111 to midpoint):
  EXPECT_FP_EQ_ALL_ROUNDING(
      0x1.d94ae07f3a803p-5, 0x1.d94ae07f3a804p-5, 0x1.d94ae07f3a803p-5,
      0x1.d94ae07f3a803p-5,
      LIBC_NAMESPACE::pow(0x1.ffe1ba8a18c71p-1, 0x1.81cp+13));
  EXPECT_FP_EQ_ALL_ROUNDING(
      0x1.8baa43f9027dfp+17, 0x1.8baa43f9027e0p+17, 0x1.8baa43f9027dfp+17,
      0x1.8baa43f9027dfp+17,
      LIBC_NAMESPACE::pow(0x1.0070e1f54c396p+0, 0x1.bbcp+12));

  // CORE-MATH 256-bit worst cases (relative distance ~2^-116 to 2^-118 from
  // 54-bit rounding boundaries, including negative base with odd integer
  // y = 4025):
  EXPECT_FP_EQ_ALL_ROUNDING(
      0x1.0000000000001p+0, 0x1.0000000000001p+0, 1.0, 1.0,
      LIBC_NAMESPACE::pow(0x1.171436ab898ccp-796, -0x1.db3136be02ca6p-63));
  EXPECT_FP_EQ_ALL_ROUNDING(
      0x1.7d646b1ee4f67p-302, 0x1.7d646b1ee4f68p-302, 0x1.7d646b1ee4f67p-302,
      0x1.7d646b1ee4f67p-302,
      LIBC_NAMESPACE::pow(0x1.c72ce7406b3cep-1, 0x1.bcp+10));
  EXPECT_FP_EQ_ALL_ROUNDING(
      0x1.b10d94bb8fd99p+863, 0x1.b10d94bb8fd99p+863, 0x1.b10d94bb8fd98p+863,
      0x1.b10d94bb8fd98p+863,
      LIBC_NAMESPACE::pow(0x1.290eb7bcc6a0ep+0, 0x1.f72p+11));
  EXPECT_FP_EQ_ALL_ROUNDING(
      -0x1.b10d94bb8fd99p+863, -0x1.b10d94bb8fd98p+863, -0x1.b10d94bb8fd99p+863,
      -0x1.b10d94bb8fd98p+863,
      LIBC_NAMESPACE::pow(-0x1.290eb7bcc6a0ep+0, 0x1.f72p+11));
}
#endif // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS

TEST_F(LlvmLibcPowTest, SubnormalBase) {
  EXPECT_FP_EQ(0x1.0p825, LIBC_NAMESPACE::pow(0x1.0p-1024, -0x1.9c8p-1));
  EXPECT_FP_EQ(0x1.0p-512, LIBC_NAMESPACE::pow(0x1.0p-1024, 0.5));
  EXPECT_FP_EQ(0x1.0p512, LIBC_NAMESPACE::pow(0x1.0p-1024, -0.5));
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
