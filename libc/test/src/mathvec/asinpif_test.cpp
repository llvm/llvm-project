//===-- Unittests for asinpif ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/asinpif.h"
#include "src/mathvec/asinpif.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecAsinpifTest = LIBC_NAMESPACE::testing::FPTest<float>;

using AsinpifOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::asinpif,
                                              LIBC_NAMESPACE::asinpif>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecAsinpifTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AsinpifOp>(aNaN));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AsinpifOp>(inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AsinpifOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AsinpifOp>(2.0f));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AsinpifOp>(-2.0f));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<AsinpifOp>(0.0f));

  EXPECT_SIMD_EQ(splat(-0.0f), wrap_vector<AsinpifOp>(-0.0f));

  EXPECT_SIMD_EQ(splat(0.5f), wrap_vector<AsinpifOp>(1.0f));

  EXPECT_SIMD_EQ(splat(-0.5f), wrap_vector<AsinpifOp>(-1.0f));
}

TEST_F(LlvmLibcVecAsinpifTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = 0x7f80'0000U / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    TEST_VARIED_CASES(x, AsinpifOp);
  }
}
