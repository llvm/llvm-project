//===-- Unittests for asinpif16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/math/asinpif16.h"
#include "test/UnitTest/FPMatcher.h"

using LlvmLibcAsinpif16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcAsinpif16Test, SpecialNumbers) {
  // zero
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::asinpif16(zero));

  // +/-1
  EXPECT_FP_EQ(0.5f16, LIBC_NAMESPACE::asinpif16(1.0));
  EXPECT_FP_EQ(-0.5f16, LIBC_NAMESPACE::asinpif16(-1.0));

  // NaN inputs
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(FPBits::quiet_nan().get_val()));

  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(FPBits::signaling_nan().get_val()));
  EXPECT_MATH_ERRNO(0);

  // infinity inputs -> should return NaN
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(), LIBC_NAMESPACE::asinpif16(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcAsinpif16Test, OutOfRange) {
  // Test values > 1
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(1.5f16));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(2.0f16));
  EXPECT_MATH_ERRNO(EDOM);

  // Test values < -1
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(-1.5f16));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(-2.0f16));
  EXPECT_MATH_ERRNO(EDOM);

  // Test maximum normal value (should be > 1 for float16)
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(FPBits::max_normal().get_val()));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcAsinpif16Test, SymmetryProperty) {
  // Test that asinpi(-x) = -asinpi(x)
  constexpr float16 TEST_VALS[] = {0.1f16, 0.25f16, 0.5f16, 0.75f16,
                                   0.9f16, 0.99f16, 1.0f16};

  for (float16 x : TEST_VALS) {
    FPBits neg_x_bits(x);
    neg_x_bits.set_sign(Sign::NEG);
    float16 neg_x = neg_x_bits.get_val();

    float16 pos_result = LIBC_NAMESPACE::asinpif16(x);
    float16 neg_result = LIBC_NAMESPACE::asinpif16(neg_x);

    EXPECT_FP_EQ(pos_result, FPBits(neg_result).abs().get_val());
  }
}
