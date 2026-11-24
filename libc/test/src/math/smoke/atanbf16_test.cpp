//===-- Unittests for atanbf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/atanbf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcAtanBf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)
public:
  void test_special_numbers() {
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atanbf16(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::atanbf16(sNaN), FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::atanbf16(zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::atanbf16(neg_zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(0x1.921fb6p0f),
                              LIBC_NAMESPACE::atanbf16(inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(-0x1.921fb6p0f),
                              LIBC_NAMESPACE::atanbf16(neg_inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(0x1.921fb6p-1f),
                              LIBC_NAMESPACE::atanbf16(bfloat16(1.0)));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(-0x1.921fb6p-1f),
                              LIBC_NAMESPACE::atanbf16(bfloat16(-1.0)));
    EXPECT_MATH_ERRNO(0);
  }
};
TEST_F(LlvmLibcAtanBf16Test, SpecialNumbers) { test_special_numbers(); }

TEST_F(LlvmLibcAtanBf16Test, NoSpuriousRangeExceptions) {
  using Bits = LIBC_NAMESPACE::fputil::FPBits<bfloat16>;
  using namespace LIBC_NAMESPACE::fputil::testing;
#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  constexpr RoundingMode MODES[] = {RoundingMode::Nearest};
#else
  constexpr RoundingMode MODES[] = {RoundingMode::Nearest, RoundingMode::Upward,
                                    RoundingMode::Downward,
                                    RoundingMode::TowardZero};
#endif // LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  // Each result is normal, including the tiny inputs that used to underflow
  // while squaring x or computing the small correction in float.
  constexpr uint16_t INPUTS[] = {0x0100, 0x2000, 0x3c00,
                                 0x4580, 0x5f80, 0x7f7f};
  for (RoundingMode mode : MODES) {
    ForceRoundingMode rounding(mode);
    if (!rounding.success)
      continue;
    for (uint16_t magnitude : INPUTS) {
      constexpr uint16_t SIGNS[] = {0, 0x8000};
      for (uint16_t sign : SIGNS) {
        bool negative = sign != 0;
        uint16_t expected = magnitude < 0x4580 ? magnitude : uint16_t(0x3fc9);
        if (magnitude < 0x4580) {
          if (mode == RoundingMode::TowardZero ||
              (negative && mode == RoundingMode::Upward) ||
              (!negative && mode == RoundingMode::Downward))
            --expected;
        } else if ((negative && mode == RoundingMode::Downward) ||
                   (!negative && mode == RoundingMode::Upward)) {
          ++expected;
        }
        LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
        bfloat16 result = LIBC_NAMESPACE::atanbf16(
            Bits(uint16_t(magnitude | sign)).get_val());
        EXPECT_FP_EQ(Bits(uint16_t(expected | sign)).get_val(), result);
        EXPECT_EQ(
            LIBC_NAMESPACE::fputil::test_except(FE_OVERFLOW | FE_UNDERFLOW), 0);
        EXPECT_MATH_ERRNO(0);
        LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      }
    }
  }
}
