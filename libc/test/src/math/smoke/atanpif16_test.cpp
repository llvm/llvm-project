//===-- Unittests for atanpif16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atanpif16.h"
#include "test/UnitTest/FPMatcher.h"

using LIBC_NAMESPACE::cpp::array;
using LlvmLibcAtanpif16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcAtanpif16Test, SpecialNumbers) {
  // zero
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::atanpif16(zero));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::atanpif16(neg_zero));

  // NaN inputs
  EXPECT_FP_EQ(FPBits::quiet_nan().get_val(),
               LIBC_NAMESPACE::atanpif16(FPBits::quiet_nan().get_val()));

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::atanpif16(aNaN));

  // infinity inputs -> should return +/-0.5
  EXPECT_FP_EQ(0.5f16, LIBC_NAMESPACE::atanpif16(inf));
  EXPECT_FP_EQ(-0.5f16, LIBC_NAMESPACE::atanpif16(neg_inf));
}

TEST_F(LlvmLibcAtanpif16Test, SymmetryProperty) {
  // Test that atanpi(-x) = -atanpi(x)
  constexpr array<float16, 12> TEST_VALS = {
      0.1f16, 0.25f16, 0.5f16,  0.75f16, 1.0f16,   1.5f16,
      2.0f16, 5.0f16,  10.0f16, 50.0f16, 100.0f16, 1000.0f16};

  for (float16 x : TEST_VALS) {
    FPBits neg_x_bits(x);
    neg_x_bits.set_sign(Sign::NEG);
    float16 neg_x = neg_x_bits.get_val();

    float16 pos_result = LIBC_NAMESPACE::atanpif16(x);
    float16 neg_result = LIBC_NAMESPACE::atanpif16(neg_x);

    EXPECT_FP_EQ(pos_result, FPBits(neg_result).abs().get_val());
  }
}

TEST_F(LlvmLibcAtanpif16Test, MonotonicityProperty) {
  // Test that atanpi is monotonically increasing
  constexpr array<float16, 15> TEST_VALS = {
      -1000.0f16, -100.0f16, -10.0f16, -2.0f16,  -1.0f16,
      -0.5f16,    -0.1f16,   0.0f16,   0.1f16,   0.5f16,
      1.0f16,     2.0f16,    10.0f16,  100.0f16, 1000.0f16};
  for (size_t i = 0; i < TEST_VALS.size() - 1; ++i) {
    float16 x1 = TEST_VALS[i];
    float16 x2 = TEST_VALS[i + 1];
    float16 result1 = LIBC_NAMESPACE::atanpif16(x1);
    float16 result2 = LIBC_NAMESPACE::atanpif16(x2);

    EXPECT_TRUE(result1 < result2);
  }
}
