//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD asin.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/asinf.h"
#include "src/mathvec/asinf.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecAsinfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using AsinfOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::asinf,
                                              LIBC_NAMESPACE::asinf>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecAsinfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AsinfOp>(aNaN));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<AsinfOp>(0.0f));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AsinfOp>(inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AsinfOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AsinfOp>(2.0f));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AsinfOp>(-2.0f));
}

TEST_F(LlvmLibcVecAsinfTest, SpecificBitPatterns) {
  constexpr int N = 11;
  constexpr uint32_t INPUTS[N] = {
      0x3f000000, // x = 0.5f
      0x3f3504f3, // x = sqrt(2)/2, FE_DOWNWARD
      0x3f3504f4, // x = sqrt(2)/2, FE_UPWARD
      0x3f5db3d7, // x = sqrt(3)/2, FE_DOWNWARD
      0x3f5db3d8, // x = sqrt(3)/2, FE_UPWARD
      0x3f800000, // x = 1.0f
      0x40000000, // x = 2.0f
      0x3d09bf86, // x = 0x1.137f0cp-5f
      0x3de5fa1e, // x = 0x1.cbf43cp-4f
      0x3f083a1a, // x = 0x1.107434p-1f
      0x3f7741b6, // x = 0x1.ee836cp-1f
  };

  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_SIMD_EQ(wrap_ref<AsinfOp>(x, -x), wrap_vector<AsinfOp>(x, -x));
  }
}

TEST_F(LlvmLibcVecAsinfTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(AsinfOp);
}
