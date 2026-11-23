//===-- Unittests for asinpif16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/__support/FPUtil/cast.h"
#include "src/math/asinpif16.h"
#include "test/UnitTest/FPMatcher.h"

using LlvmLibcAsinpif16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcAsinpif16Test, SpecialNumbers) {
  using LIBC_NAMESPACE::fputil::cast;
  // zero
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::asinpif16(zero));

  // +/-1
  EXPECT_FP_EQ(cast<float16>(0.5f),
               LIBC_NAMESPACE::asinpif16(cast<float16>(1.0f)));
  EXPECT_FP_EQ(cast<float16>(-0.5f),
               LIBC_NAMESPACE::asinpif16(cast<float16>(-1.0f)));

  // NaN inputs
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::asinpif16(aNaN));

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::asinpif16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  // infinity inputs -> should return NaN
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::asinpif16(inf), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::asinpif16(neg_inf),
                                  FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcAsinpif16Test, OutOfRange) {
  using LIBC_NAMESPACE::fputil::cast;
  // Test values > 1
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::asinpif16(cast<float16>(1.5f)));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::asinpif16(cast<float16>(2.0f)));
  EXPECT_MATH_ERRNO(EDOM);

  // Test values < -1
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::asinpif16(cast<float16>(-1.5f)));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::asinpif16(cast<float16>(-2.0f)));
  EXPECT_MATH_ERRNO(EDOM);

  // Test maximum normal value (should be > 1 for float16)
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::asinpif16(FPBits::max_normal().get_val()));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcAsinpif16Test, SymmetryProperty) {
  using LIBC_NAMESPACE::fputil::cast;
  // Test that asinpi(-x) = -asinpi(x)
  constexpr float16 TEST_VALS[] = {cast<float16>(0.1f), cast<float16>(0.25f),
                                   cast<float16>(0.5f), cast<float16>(0.75f),
                                   cast<float16>(0.9f), cast<float16>(0.99f),
                                   cast<float16>(1.0f)};

  for (float16 x : TEST_VALS) {
    FPBits neg_x_bits(x);
    neg_x_bits.set_sign(Sign::NEG);
    float16 neg_x = neg_x_bits.get_val();

    float16 pos_result = LIBC_NAMESPACE::asinpif16(x);
    float16 neg_result = LIBC_NAMESPACE::asinpif16(neg_x);

    EXPECT_FP_EQ(pos_result, FPBits(neg_result).abs().get_val());
  }
}
