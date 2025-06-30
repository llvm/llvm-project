//===-- Unittests for asinpif16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/asinpif16.h"
#include "src/math/fabs.h"
#include "test/UnitTest/FPMatcher.h"

using LlvmLibcAsinpif16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcAsinpif16Test, SpecialNumbers) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float16>;

  // zero
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::asinpif16(0.0f16));

  // +/-1
  EXPECT_FP_EQ(float16(0.5), LIBC_NAMESPACE::asinpif16(float16(1.0)));
  EXPECT_FP_EQ(float16(-0.5), LIBC_NAMESPACE::asinpif16(float16(-1.0)));

  // NaN inputs
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(FPBits::quiet_nan().get_val()));

  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(FPBits::signaling_nan().get_val()));

  // infinity inputs -> should return NaN
  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(), LIBC_NAMESPACE::asinpif16(inf));
  EXPECT_MATH_ERRNO(EDOM);

  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcAsinpif16Test, OutOfRange) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float16>;

  // Test values > 1
  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(float16(1.5)));
  EXPECT_MATH_ERRNO(EDOM);

  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(float16(2.0)));
  EXPECT_MATH_ERRNO(EDOM);

  // Test values < -1
  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(float16(-1.5)));
  EXPECT_MATH_ERRNO(EDOM);

  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(float16(-2.0)));
  EXPECT_MATH_ERRNO(EDOM);

  // Test maximum normal value (should be > 1 for float16)
  errno = 0;
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::asinpif16(FPBits::max_normal().get_val()));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcAsinpif16Test, SymmetryProperty) {
  // Test that asinpi(-x) = -asinpi(x)
  constexpr float16 test_vals[] = {0.1f16, 0.25f16, 0.5f16, 0.75f16,
                                   0.9f16, 0.99f16, 1.0f16};

  for (float16 x : test_vals) {
    if (x <= 1.0) {
      float16 pos_result = LIBC_NAMESPACE::asinpif16(x);
      float16 neg_result = LIBC_NAMESPACE::asinpif16(-x);

      EXPECT_FP_EQ(pos_result,
                   static_cast<float16>(LIBC_NAMESPACE::fabs(neg_result)));
    }
  }
}
