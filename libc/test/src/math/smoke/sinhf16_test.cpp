//===-- Unittests for sinhf16 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/cast.h"
#include "src/math/sinhf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcSinhf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcSinhf16Test, AllSubnormalRounding) {
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
            (negative && mode == RoundingMode::Downward))
          ++expected;
        bool tiny = (expected & 0x7fff) < 0x0400;
        EXPECT_FP_EQ_WITH_EXCEPTION(
            Bits(expected).get_val(),
            LIBC_NAMESPACE::sinhf16(Bits(uint16_t(magnitude | sign)).get_val()),
            FE_INEXACT | (tiny ? FE_UNDERFLOW : 0));
        EXPECT_MATH_ERRNO(tiny ? ERANGE : 0);
      }
    }
  }
}

TEST_F(LlvmLibcSinhf16Test, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::sinhf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::sinhf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::sinhf16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, LIBC_NAMESPACE::sinhf16(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::sinhf16(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::sinhf16(neg_zero));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcSinhf16Test, Overflow) {
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::sinhf16(max_normal),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::sinhf16(neg_max_normal),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  // round(asinh(2^16), HP, RU);
  float16 x = LIBC_NAMESPACE::fputil::cast<float16>(0x1.794p+3);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(inf, LIBC_NAMESPACE::sinhf16(x),
                                               FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(inf, LIBC_NAMESPACE::sinhf16(x),
                                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      max_normal, LIBC_NAMESPACE::sinhf16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      max_normal, LIBC_NAMESPACE::sinhf16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  // round(asinh(-2^16), HP, RD);
  x = LIBC_NAMESPACE::fputil::cast<float16>(-0x1.794p+3);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      neg_inf, LIBC_NAMESPACE::sinhf16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      neg_max_normal, LIBC_NAMESPACE::sinhf16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      neg_inf, LIBC_NAMESPACE::sinhf16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      neg_max_normal, LIBC_NAMESPACE::sinhf16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcSinhf16Test, OverflowStatusAllRoundingModes) {
  using Bits = LIBC_NAMESPACE::fputil::FPBits<float16>;
  const float16 x = Bits(uint16_t(0x5000)).get_val();
  const float16 largest = Bits(uint16_t(0x7bff)).get_val();
  const float16 infinity = Bits::inf().get_val();
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      infinity, LIBC_NAMESPACE::sinhf16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      infinity, LIBC_NAMESPACE::sinhf16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      largest, LIBC_NAMESPACE::sinhf16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      largest, LIBC_NAMESPACE::sinhf16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
#endif // !LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY

#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  const float16 negative = Bits(uint16_t(0xd000)).get_val();
  const float16 lowest = Bits(uint16_t(0xfbff)).get_val();
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      lowest, LIBC_NAMESPACE::sinhf16(negative), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      lowest, LIBC_NAMESPACE::sinhf16(negative), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
#endif // !LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
}
