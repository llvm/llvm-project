//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/tanhbf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcTanhBf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

TEST_F(LlvmLibcTanhBf16Test, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::tanhbf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::tanhbf16(sNaN),
                                           FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(1.0f), LIBC_NAMESPACE::tanhbf16(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(-1.0f), LIBC_NAMESPACE::tanhbf16(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::tanhbf16(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::tanhbf16(neg_zero));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcTanhBf16Test, SmallAndLargeValues) {
  bfloat16 x = FPBits(uint16_t(0x3d80U)).get_val(); // 0.0625
  bfloat16 below_x = FPBits(uint16_t(0x3d7fU)).get_val();
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(x, LIBC_NAMESPACE::tanhbf16(x),
                                               FE_INEXACT);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      below_x, LIBC_NAMESPACE::tanhbf16(x), FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      bfloat16(1.0f), LIBC_NAMESPACE::tanhbf16(max_normal), FE_INEXACT);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      FPBits(uint16_t(0x3f7fU)).get_val(), LIBC_NAMESPACE::tanhbf16(max_normal),
      FE_INEXACT);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      FPBits(uint16_t(0xbf7fU)).get_val(),
      LIBC_NAMESPACE::tanhbf16(neg_max_normal), FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcTanhBf16Test, TinyValues) {
  bfloat16 smallest = FPBits(uint16_t(0x0001U)).get_val();
  bfloat16 neg_smallest = FPBits(uint16_t(0x8001U)).get_val();
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      smallest, LIBC_NAMESPACE::tanhbf16(smallest), FE_UNDERFLOW | FE_INEXACT);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      zero, LIBC_NAMESPACE::tanhbf16(smallest), FE_UNDERFLOW | FE_INEXACT);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      neg_zero, LIBC_NAMESPACE::tanhbf16(neg_smallest),
      FE_UNDERFLOW | FE_INEXACT);

  bfloat16 min_normal = FPBits(uint16_t(0x0080U)).get_val();
  bfloat16 max_subnormal = FPBits(uint16_t(0x007fU)).get_val();
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      max_subnormal, LIBC_NAMESPACE::tanhbf16(min_normal),
      FE_UNDERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
}
