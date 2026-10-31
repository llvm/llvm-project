//===-- Unittests for expm1f16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/cast.h"
#include "src/math/expm1f16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExpm1f16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcExpm1f16Test, AllSubnormalRounding) {
  using Bits = LIBC_NAMESPACE::fputil::FPBits<float16>;
  using namespace LIBC_NAMESPACE::fputil::testing;
#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  constexpr RoundingMode MODES[] = {RoundingMode::Nearest};
#else
  constexpr RoundingMode MODES[] = {RoundingMode::Nearest, RoundingMode::Upward,
                                    RoundingMode::Downward,
                                    RoundingMode::TowardZero};
#endif
  for (RoundingMode mode : MODES) {
    ForceRoundingMode rounding(mode);
    if (!rounding.success)
      continue;
    for (uint16_t magnitude = 1; magnitude < 0x0400; ++magnitude) {
      constexpr uint16_t SIGNS[] = {0, 0x8000};
      for (uint16_t sign : SIGNS) {
        bool negative = sign != 0;
        uint16_t expected = static_cast<uint16_t>(magnitude | sign);
        if ((!negative && mode == RoundingMode::Upward) ||
            (negative && (mode == RoundingMode::Upward ||
                          mode == RoundingMode::TowardZero)))
          expected = static_cast<uint16_t>(expected + (negative ? -1 : 1));
        bool tiny = (expected & 0x7fff) < 0x0400;
        EXPECT_FP_EQ_WITH_EXCEPTION(
            Bits(expected).get_val(),
            LIBC_NAMESPACE::expm1f16(
                Bits(uint16_t(magnitude | sign)).get_val()),
            FE_INEXACT | (tiny ? FE_UNDERFLOW : 0));
        EXPECT_MATH_ERRNO(tiny ? ERANGE : 0);
      }
    }
  }
}

TEST_F(LlvmLibcExpm1f16Test, NegativeMinNormalTininessAfterRounding) {
  using Bits = LIBC_NAMESPACE::fputil::FPBits<float16>;
  using namespace LIBC_NAMESPACE::fputil::testing;
#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  constexpr RoundingMode MODES[] = {RoundingMode::Nearest};
#else
  constexpr RoundingMode MODES[] = {RoundingMode::Nearest, RoundingMode::Upward,
                                    RoundingMode::Downward,
                                    RoundingMode::TowardZero};
#endif
  for (RoundingMode mode : MODES) {
    ForceRoundingMode rounding(mode);
    if (!rounding.success)
      continue;
    uint16_t expected = 0x8400;
    if (mode == RoundingMode::Upward || mode == RoundingMode::TowardZero)
      expected = 0x83ff;
    bool tiny = (expected & 0x7fff) < 0x0400;
    EXPECT_FP_EQ_WITH_EXCEPTION(
        Bits(expected).get_val(),
        LIBC_NAMESPACE::expm1f16(Bits(uint16_t(0x8400)).get_val()),
        FE_INEXACT | (tiny ? FE_UNDERFLOW : 0));
    EXPECT_MATH_ERRNO(tiny ? ERANGE : 0);
  }
}

TEST_F(LlvmLibcExpm1f16Test, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::expm1f16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::expm1f16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::expm1f16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(LIBC_NAMESPACE::fputil::cast<float16>(-1.0),
                            LIBC_NAMESPACE::expm1f16(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::expm1f16(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::expm1f16(neg_zero));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExpm1f16Test, Overflow) {
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::expm1f16(max_normal),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  // round(16 * log(2), HP, RN);
  float16 x = LIBC_NAMESPACE::fputil::cast<float16>(0x1.63p+3);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(inf, LIBC_NAMESPACE::expm1f16(x),
                                               FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(inf, LIBC_NAMESPACE::expm1f16(x),
                                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      max_normal, LIBC_NAMESPACE::expm1f16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      max_normal, LIBC_NAMESPACE::expm1f16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExpm1f16Test, ResultNearNegOne) {
  EXPECT_FP_EQ_WITH_EXCEPTION(LIBC_NAMESPACE::fputil::cast<float16>(-1.0),
                              LIBC_NAMESPACE::expm1f16(neg_max_normal),
                              FE_INEXACT);

  // round(-11 * log(2), HP, RN);
  float16 x = LIBC_NAMESPACE::fputil::cast<float16>(-0x1.e8p+2);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      LIBC_NAMESPACE::fputil::cast<float16>(-0x1.ffcp-1),
      LIBC_NAMESPACE::expm1f16(x), FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      LIBC_NAMESPACE::fputil::cast<float16>(-0x1.ffcp-1),
      LIBC_NAMESPACE::expm1f16(x), FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      LIBC_NAMESPACE::fputil::cast<float16>(-1.0), LIBC_NAMESPACE::expm1f16(x),
      FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      LIBC_NAMESPACE::fputil::cast<float16>(-0x1.ffcp-1),
      LIBC_NAMESPACE::expm1f16(x), FE_INEXACT);

  x = LIBC_NAMESPACE::fputil::cast<float16>(-0x1.0a4p+3);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      LIBC_NAMESPACE::fputil::cast<float16>(-1.0), LIBC_NAMESPACE::expm1f16(x),
      FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      LIBC_NAMESPACE::fputil::cast<float16>(-0x1.ffcp-1),
      LIBC_NAMESPACE::expm1f16(x), FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      LIBC_NAMESPACE::fputil::cast<float16>(-1.0), LIBC_NAMESPACE::expm1f16(x),
      FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      LIBC_NAMESPACE::fputil::cast<float16>(-0x1.ffcp-1),
      LIBC_NAMESPACE::expm1f16(x), FE_INEXACT);
}
