//===-- Unittests for asinpif16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/math/asinpif16.h"
#include "test/UnitTest/FPMatcher.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcAsinpif16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcAsinpif16Test, SpecialNumbers) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float16>;

  // Test zero
  EXPECT_FP_EQ(FPBits::zero(), LIBC_NAMESPACE::asinpif16(FPBits::zero()));
  EXPECT_FP_EQ(FPBits::neg_zero(),
               LIBC_NAMESPACE::asinpif16(FPBits::neg_zero()));

  // Test +/-1
  EXPECT_FP_EQ(float16(0.5), LIBC_NAMESPACE::asinpif16(float16(1.0)));
  EXPECT_FP_EQ(float16(-0.5), LIBC_NAMESPACE::asinpif16(float16(-1.0)));

  // Test NaN inputs
  EXPECT_FP_EQ(FPBits::quiet_nan(),
               LIBC_NAMESPACE::asinpif16(FPBits::quiet_nan()));
  EXPECT_FP_EQ(FPBits::quiet_nan(),
               LIBC_NAMESPACE::asinpif16(FPBits::signaling_nan()));

  // Test infinity inputs - should return NaN and set errno
  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan(), LIBC_NAMESPACE::asinpif16(FPBits::inf()));
  EXPECT_MATH_ERRNO(EDOM);

  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan(),
               LIBC_NAMESPACE::asinpif16(FPBits::neg_inf()));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcAsinpif16Test, OutOfRange) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float16>;

  // Test values > 1
  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan(), LIBC_NAMESPACE::asinpif16(float16(1.5)));
  EXPECT_MATH_ERRNO(EDOM);

  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan(), LIBC_NAMESPACE::asinpif16(float16(2.0)));
  EXPECT_MATH_ERRNO(EDOM);

  // Test values < -1
  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan(), LIBC_NAMESPACE::asinpif16(float16(-1.5)));
  EXPECT_MATH_ERRNO(EDOM);

  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan(), LIBC_NAMESPACE::asinpif16(float16(-2.0)));
  EXPECT_MATH_ERRNO(EDOM);

  // Test maximum normal value (should be > 1 for float16)
  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan(),
               LIBC_NAMESPACE::asinpif16(FPBits::max_normal()));
  EXPECT_MATH_ERRNO(EDOM);

  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan(),
               LIBC_NAMESPACE::asinpif16(FPBits::max_normal().get_sign()));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcAsinpif16Test, SmallValues) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float16>;

  // Test very small values - should be close to x/π
  constexpr float16 small_vals[] = {
      0x1.0p-10,  // Small positive
      -0x1.0p-10, // Small negative
      0x1.0p-14,  // Very small positive
      -0x1.0p-14, // Very small negative
  };

  for (float16 x : small_vals) {
    float16 result = LIBC_NAMESPACE::asinpif16(x);
    // For small x, asinpi(x) ≈ x/π ≈ 0.318309886 * x
    // We expect the result to be close to x/π but not exactly due to polynomial
    // approximation
    EXPECT_TRUE(LIBC_NAMESPACE::fputil::abs(result) <=
                LIBC_NAMESPACE::fputil::abs(x));
  }

  // Test minimum subnormal values
  EXPECT_FP_EQ_ALL_ROUNDING(FPBits::min_subnormal(),
                            LIBC_NAMESPACE::asinpif16(FPBits::min_subnormal()));
  EXPECT_FP_EQ_ALL_ROUNDING(
      FPBits::min_subnormal().get_sign(),
      LIBC_NAMESPACE::asinpif16(FPBits::min_subnormal().get_sign()));
}


TEST_F(LlvmLibcAsinpif16Test, SymmetryProperty) {
  // Test that asinpi(-x) = -asinpi(x)
  constexpr float16 test_vals[] = {0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0};

  for (float16 x : test_vals) {
    if (x <= 1.0) { // Only test valid domain
      float16 pos_result = LIBC_NAMESPACE::asinpif16(x);
      float16 neg_result = LIBC_NAMESPACE::asinpif16(-x);

      EXPECT_FP_EQ(pos_result, -neg_result);
    }
  }
}

TEST_F(LlvmLibcAsinpif16Test, RangeValidation) {
  // Test that output is always in [-0.5, 0.5] for valid inputs
  constexpr int num_tests = 1000;

  for (int i = 0; i <= num_tests; ++i) {
    // Generate test value in [-1, 1]
    float t = -1.0f + (2.0f * i) / num_tests;
    float16 x = static_cast<float16>(t);

    if (LIBC_NAMESPACE::fputil::abs(x) <= 1.0) {
      float16 result = LIBC_NAMESPACE::asinpif16(x);

      // Result should be in [-0.5, 0.5]
      EXPECT_TRUE(result >= -0.5);
      EXPECT_TRUE(result <= 0.5);
    }
  }
}