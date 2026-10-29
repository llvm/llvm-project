//===-- Unittests for exp2m1f16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/cast.h"
#include "src/math/exp2m1f16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExp2m1f16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
TEST_F(LlvmLibcExp2m1f16Test, SubnormalBoundary) {
  using Bits = LIBC_NAMESPACE::fputil::FPBits<float16>;
  const float16 x = Bits(uint16_t(0x8461)).get_val();
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(Bits(uint16_t(0x8309)).get_val(),
                                               LIBC_NAMESPACE::exp2m1f16(x),
                                               FE_UNDERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(Bits(uint16_t(0x8308)).get_val(),
                                              LIBC_NAMESPACE::exp2m1f16(x),
                                              FE_UNDERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      Bits(uint16_t(0x8309)).get_val(), LIBC_NAMESPACE::exp2m1f16(x),
      FE_UNDERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      Bits(uint16_t(0x8308)).get_val(), LIBC_NAMESPACE::exp2m1f16(x),
      FE_UNDERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);
#endif
}
#endif

TEST_F(LlvmLibcExp2m1f16Test, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::exp2m1f16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::exp2m1f16(sNaN),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::exp2m1f16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(-1.0, LIBC_NAMESPACE::exp2m1f16(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::exp2m1f16(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::exp2m1f16(neg_zero));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExp2m1f16Test, Overflow) {
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::exp2m1f16(max_normal),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  float16 x = LIBC_NAMESPACE::fputil::cast<float16>(16.0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      inf, LIBC_NAMESPACE::exp2m1f16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(inf, LIBC_NAMESPACE::exp2m1f16(x),
                                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      max_normal, LIBC_NAMESPACE::exp2m1f16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      max_normal, LIBC_NAMESPACE::exp2m1f16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExp2m1f16Test, ResultNearNegOne) {
  EXPECT_FP_EQ_WITH_EXCEPTION(-1.0, LIBC_NAMESPACE::exp2m1f16(neg_max_normal),
                              FE_INEXACT);

  EXPECT_FP_EQ_ALL_ROUNDING(
      -0x1.ffcp-1,
      LIBC_NAMESPACE::exp2m1f16(LIBC_NAMESPACE::fputil::cast<float16>(-11.0)));

  float16 x = LIBC_NAMESPACE::fputil::cast<float16>(-12.0);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      -1.0, LIBC_NAMESPACE::exp2m1f16(x), FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      -0x1.ffcp-1, LIBC_NAMESPACE::exp2m1f16(x), FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      -1.0, LIBC_NAMESPACE::exp2m1f16(x), FE_INEXACT);

  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      -0x1.ffcp-1, LIBC_NAMESPACE::exp2m1f16(x), FE_INEXACT);
}

TEST_F(LlvmLibcExp2m1f16Test, OverflowStatusAllRoundingModes) {
  using Bits = LIBC_NAMESPACE::fputil::FPBits<float16>;
  const float16 x = Bits(uint16_t(0x5000)).get_val();
  const float16 largest = Bits(uint16_t(0x7bff)).get_val();
  const float16 infinity = Bits::inf().get_val();
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      infinity, LIBC_NAMESPACE::exp2m1f16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      infinity, LIBC_NAMESPACE::exp2m1f16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      largest, LIBC_NAMESPACE::exp2m1f16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      largest, LIBC_NAMESPACE::exp2m1f16(x), FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(0);
#endif // !LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
}
