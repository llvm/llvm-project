//===-- Unittests for expbf16 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/math/expbf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"

using LlvmLibcExpbf16RangeTest = LIBC_NAMESPACE::testing::FEnvSafeTest;

TEST_F(LlvmLibcExpbf16RangeTest, BfloatRangeEdges) {
  using Bits = LIBC_NAMESPACE::fputil::FPBits<bfloat16>;
  using namespace LIBC_NAMESPACE::fputil::testing;
  constexpr uint16_t INPUTS[] = {0x0001, 0x0080, 0x0100, 0x3980};
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
    for (uint16_t magnitude : INPUTS) {
      constexpr uint16_t SIGNS[] = {0, 0x8000};
      for (uint16_t sign : SIGNS) {
        uint16_t expected = 0x3f80;
        if (sign == 0 && mode == RoundingMode::Upward)
          ++expected;
        if (sign != 0 && (mode == RoundingMode::Downward ||
                          mode == RoundingMode::TowardZero))
          --expected;
        LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
        bfloat16 result =
            LIBC_NAMESPACE::expbf16(Bits(uint16_t(magnitude | sign)).get_val());
        EXPECT_FP_EQ(Bits(expected).get_val(), result);
        EXPECT_EQ(
            LIBC_NAMESPACE::fputil::test_except(FE_OVERFLOW | FE_UNDERFLOW), 0);
        EXPECT_MATH_ERRNO(0);
        LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      }
    }
  }
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      Bits::min_subnormal().get_val(),
      LIBC_NAMESPACE::expbf16(Bits(uint16_t(0xc2b9)).get_val()),
      FE_UNDERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  constexpr uint16_t FAR_INPUTS[] = {0xc2ba, 0xc300, 0xff7f};
  for (uint16_t bits : FAR_INPUTS) {
    EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
        Bits::min_subnormal().get_val(),
        LIBC_NAMESPACE::expbf16(Bits(bits).get_val()),
        FE_UNDERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);
  }
#endif
}

class LlvmLibcExpbf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)
public:
  void test_special_numbers() {
    EXPECT_FP_IS_NAN(LIBC_NAMESPACE::expbf16(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::expbf16(sNaN), FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::expbf16(inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::expbf16(neg_inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(LIBC_NAMESPACE::fputil::cast<bfloat16>(1.0f),
                              LIBC_NAMESPACE::expbf16(zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(LIBC_NAMESPACE::fputil::cast<bfloat16>(1.0f),
                              LIBC_NAMESPACE::expbf16(neg_zero));
    EXPECT_MATH_ERRNO(0);
  }

  void test_overflow() {
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::expbf16(max_normal),
                                FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf,
        LIBC_NAMESPACE::expbf16(LIBC_NAMESPACE::fputil::cast<bfloat16>(89.0f)),
        FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);
  }

  void test_underflow() {
    EXPECT_FP_EQ_WITH_EXCEPTION(zero, LIBC_NAMESPACE::expbf16(neg_max_normal),
                                FE_UNDERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION(
        zero,
        LIBC_NAMESPACE::expbf16(LIBC_NAMESPACE::fputil::cast<bfloat16>(-93.0f)),
        FE_UNDERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);
  }
};

TEST_F(LlvmLibcExpbf16Test, SpecialNumbers) { test_special_numbers(); }
TEST_F(LlvmLibcExpbf16Test, Overflow) { test_overflow(); }
TEST_F(LlvmLibcExpbf16Test, Undeflow) { test_underflow(); }
