//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD acos.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/acosf.h"
#include "src/mathvec/acosf.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecAcosfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using AcosfOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::acosf,
                                              LIBC_NAMESPACE::acosf>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecAcosfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcosfOp>(aNaN));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcosfOp>(inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcosfOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(zero), wrap_vector<AcosfOp>(1.0f));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcosfOp>(2.0f));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcosfOp>(-2.0f));
}

TEST_F(LlvmLibcVecAcosfTest, SpecificBitPatterns) {
  constexpr int N = 13;
  constexpr uint32_t INPUTS[N] = {
      0x3f000000, // x = 0.5f
      0x3f3504f3, // x = sqrt(2)/2, FE_DOWNWARD
      0x3f3504f4, // x = sqrt(2)/2, FE_UPWARD
      0x3f5db3d7, // x = sqrt(3)/2, FE_DOWNWARD
      0x3f5db3d8, // x = sqrt(3)/2, FE_UPWARD
      0x3f800000, // x = 1.0f
      0x40000000, // x = 2.0f
      0x328885a3, // x = 0x1.110b46p-26
      0x39826222, // x = 0x1.04c444p-12
      0x3d09bf86, // x = 0x1.137f0cp-5f
      0x3de5fa1e, // x = 0x1.cbf43cp-4f
      0x3f083a1a, // x = 0x1.107434p-1f
      0x3f7741b6, // x = 0x1.ee836cp-1f
  };
  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_SIMD_EQ(wrap_ref<AcosfOp>(x, -x), wrap_vector<AcosfOp>(x, -x));
  }
}

TEST_F(LlvmLibcVecAcosfTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(AcosfOp);
}
