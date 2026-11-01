//===-- Unittests for acosbf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/acosbf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcAcosBf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)
public:
  void test_special_numbers() {
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acosbf16(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acosbf16(sNaN), FE_INVALID);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acosbf16(inf), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(
        aNaN, LIBC_NAMESPACE::acosbf16(neg_inf), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(0x1.921fb6p0f),
                              LIBC_NAMESPACE::acosbf16(zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(0x1.921fb6p0f),
                              LIBC_NAMESPACE::acosbf16(neg_zero));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::acosbf16(bfloat16(1.0)));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(0x1.921fb6p1f),
                              LIBC_NAMESPACE::acosbf16(bfloat16(-1.0)));
    EXPECT_MATH_ERRNO(0);
  }
};
TEST_F(LlvmLibcAcosBf16Test, SpecialNumbers) { test_special_numbers(); }

TEST_F(LlvmLibcAcosBf16Test, SmallInputs) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<bfloat16>;
  auto acos_without_underflow = [](bfloat16 x) {
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    bfloat16 result = LIBC_NAMESPACE::acosbf16(x);
    EXPECT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_UNDERFLOW), 0);
    EXPECT_MATH_ERRNO(0);
    return result;
  };

  // Include subnormals, an input that caused an intermediate underflow in
  // the float polynomial without FMA, and the small-input branch boundary.
  const uint16_t inputs[] = {0x0001, 0x007f, 0x0080, 0x209d,
                             0x397f, 0x3980, 0x3981};
  const bfloat16 pi_2_lo = FPBits(uint16_t(0x3fc9)).get_val();
  const bfloat16 pi_2_hi = FPBits(uint16_t(0x3fca)).get_val();
  for (uint16_t bits : inputs) {
    bfloat16 x = FPBits(bits).get_val();
    EXPECT_FP_EQ_ALL_ROUNDING(pi_2_lo, pi_2_hi, pi_2_lo, pi_2_lo,
                              acos_without_underflow(x));
    EXPECT_FP_EQ_ALL_ROUNDING(pi_2_lo, pi_2_hi, pi_2_lo, pi_2_lo,
                              acos_without_underflow(-x));
  }
}
