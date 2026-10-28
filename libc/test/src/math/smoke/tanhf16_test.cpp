//===-- Unittests for tanhf16 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/cast.h"
#include "src/math/tanhf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcTanhf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcTanhf16Test, AllSubnormalRounding) {
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
        if ((!negative && (mode == RoundingMode::Downward ||
                           mode == RoundingMode::TowardZero)) ||
            (negative && (mode == RoundingMode::Upward ||
                          mode == RoundingMode::TowardZero)))
          --expected;
        bool tiny = (expected & 0x7fff) < 0x0400;
        EXPECT_FP_EQ_WITH_EXCEPTION(
            Bits(expected).get_val(),
            LIBC_NAMESPACE::tanhf16(Bits(uint16_t(magnitude | sign)).get_val()),
            FE_INEXACT | (tiny ? FE_UNDERFLOW : 0));
        EXPECT_MATH_ERRNO(tiny ? ERANGE : 0);
      }
    }
  }
}

TEST_F(LlvmLibcTanhf16Test, MinNormalTininessAfterRounding) {
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
    constexpr uint16_t SIGNS[] = {0, 0x8000};
    for (uint16_t sign : SIGNS) {
      bool negative = sign != 0;
      uint16_t expected = static_cast<uint16_t>(0x0400 | sign);
      if ((!negative && (mode == RoundingMode::Downward ||
                         mode == RoundingMode::TowardZero)) ||
          (negative &&
           (mode == RoundingMode::Upward || mode == RoundingMode::TowardZero)))
        --expected;
      bool tiny = (expected & 0x7fff) < 0x0400;
      EXPECT_FP_EQ_WITH_EXCEPTION(
          Bits(expected).get_val(),
          LIBC_NAMESPACE::tanhf16(Bits(uint16_t(0x0400 | sign)).get_val()),
          FE_INEXACT | (tiny ? FE_UNDERFLOW : 0));
      EXPECT_MATH_ERRNO(tiny ? ERANGE : 0);
    }
  }
}

TEST_F(LlvmLibcTanhf16Test, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::tanhf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::tanhf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(LIBC_NAMESPACE::fputil::cast<float16>(1.0),
                            LIBC_NAMESPACE::tanhf16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(LIBC_NAMESPACE::fputil::cast<float16>(-1.0),
                            LIBC_NAMESPACE::tanhf16(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::tanhf16(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::tanhf16(neg_zero));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcTanhf16Test, ResultNearBounds) {
  EXPECT_FP_EQ_WITH_EXCEPTION(LIBC_NAMESPACE::fputil::cast<float16>(1.0),
                              LIBC_NAMESPACE::tanhf16(max_normal), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(LIBC_NAMESPACE::fputil::cast<float16>(-1.0),
                              LIBC_NAMESPACE::tanhf16(neg_max_normal),
                              FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  // round(atanh(1 - 2^-11), HP, RU);
  float16 x = LIBC_NAMESPACE::fputil::cast<float16>(0x1.0a4p+2);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      LIBC_NAMESPACE::fputil::cast<float16>(0x1.ffcp-1),
      LIBC_NAMESPACE::tanhf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      LIBC_NAMESPACE::fputil::cast<float16>(1.0), LIBC_NAMESPACE::tanhf16(x),
      FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      LIBC_NAMESPACE::fputil::cast<float16>(0x1.ffcp-1),
      LIBC_NAMESPACE::tanhf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      LIBC_NAMESPACE::fputil::cast<float16>(0x1.ffcp-1),
      LIBC_NAMESPACE::tanhf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  x = LIBC_NAMESPACE::fputil::cast<float16>(0x1.208p+2);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      LIBC_NAMESPACE::fputil::cast<float16>(1.0), LIBC_NAMESPACE::tanhf16(x),
      FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      LIBC_NAMESPACE::fputil::cast<float16>(1.0), LIBC_NAMESPACE::tanhf16(x),
      FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      LIBC_NAMESPACE::fputil::cast<float16>(0x1.ffcp-1),
      LIBC_NAMESPACE::tanhf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      LIBC_NAMESPACE::fputil::cast<float16>(0x1.ffcp-1),
      LIBC_NAMESPACE::tanhf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  // round(atanh(-1 + 2^-11), HP, RD);
  x = LIBC_NAMESPACE::fputil::cast<float16>(-0x1.0a4p+2);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      LIBC_NAMESPACE::fputil::cast<float16>(-0x1.ffcp-1),
      LIBC_NAMESPACE::tanhf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      LIBC_NAMESPACE::fputil::cast<float16>(-0x1.ffcp-1),
      LIBC_NAMESPACE::tanhf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      LIBC_NAMESPACE::fputil::cast<float16>(-1.0), LIBC_NAMESPACE::tanhf16(x),
      FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      LIBC_NAMESPACE::fputil::cast<float16>(-0x1.ffcp-1),
      LIBC_NAMESPACE::tanhf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  x = LIBC_NAMESPACE::fputil::cast<float16>(-0x1.208p+2);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      LIBC_NAMESPACE::fputil::cast<float16>(-1.0), LIBC_NAMESPACE::tanhf16(x),
      FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      LIBC_NAMESPACE::fputil::cast<float16>(-0x1.ffcp-1),
      LIBC_NAMESPACE::tanhf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      LIBC_NAMESPACE::fputil::cast<float16>(-1.0), LIBC_NAMESPACE::tanhf16(x),
      FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      LIBC_NAMESPACE::fputil::cast<float16>(-0x1.ffcp-1),
      LIBC_NAMESPACE::tanhf16(x), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
}
