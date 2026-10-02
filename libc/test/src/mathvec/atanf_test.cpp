//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD atan.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/atanf.h"
#include "src/mathvec/atanf.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecAtanfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using AtanfOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::atanf,
                                              LIBC_NAMESPACE::atanf>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecAtanfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AtanfOp>(aNaN));

  EXPECT_SIMD_EQ(splat(0x1.921fb6p0), wrap_vector<AtanfOp>(inf));

  EXPECT_SIMD_EQ(splat(-0x1.921fb6p0), wrap_vector<AtanfOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<AtanfOp>(0.0f));
}

TEST_F(LlvmLibcVecAtanfTest, SpecificBitPatterns) {
  constexpr int N = 10;
  constexpr uint32_t INPUTS[N] = {
      0x3d8d'6b23U, // x = 0x1.1ad646p-4f
      0x3dbb'6ac7U, // x = 0x1.76d58ep-4f
      0x3fee'fcfbU, // x = 0x1.ddf9f6p+0f
      0x3ffe'2ec1U, // x = 0x1.fc5d82p+0f
      0xbd8d'6b23U, // x = -0x1.1ad646p-4f
      0xbdbb'6ac7U, // x = -0x1.76d58ep-4f
      0xbfee'fcfbU, // x = -0x1.ddf9f6p+0f
      0xbffe'2ec1U, // x = -0x1.fc5d82p+0
      0x7f80'0000U, // x = +Inf
      0xff80'0000U, // x = -Inf
  };

  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_SIMD_EQ(wrap_ref<AtanfOp>(x, -x), wrap_vector<AtanfOp>(x, -x));
  }
}

TEST_F(LlvmLibcVecAtanfTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(AtanfOp);
}
