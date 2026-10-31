//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD erf.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/erff.h"
#include "src/mathvec/erff.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecErffTest = LIBC_NAMESPACE::testing::FPTest<float>;

using ErffOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::erff,
                                              LIBC_NAMESPACE::erff>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecErffTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<ErffOp>(aNaN));

  EXPECT_SIMD_EQ(splat(1.0f), wrap_vector<ErffOp>(inf));

  EXPECT_SIMD_EQ(splat(-1.0f), wrap_vector<ErffOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<ErffOp>(0.0f));
}

TEST_F(LlvmLibcVecErffTest, TrickyInputs) {
  constexpr int N = 2;
  constexpr uint32_t INPUTS[N] = {
      0x3f65'9229U, // |x| = 0x1.cb2452p-1f
      0x4004'1e6aU, // |x| = 0x1.083cd4p+1f
  };

  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_SIMD_EQ(wrap_ref<ErffOp>(x, -x), wrap_vector<ErffOp>(x, -x));
  }
}

TEST_F(LlvmLibcVecErffTest, InFloatRange) { TEST_MATHVEC_FLOAT_RANGE(ErffOp); }
