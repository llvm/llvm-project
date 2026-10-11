//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD log2.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/log2f.h"
#include "src/mathvec/log2f.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecLog2fTest = LIBC_NAMESPACE::testing::FPTest<float>;

using Log2fOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::log2f,
                                              LIBC_NAMESPACE::log2f>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecLog2fTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Log2fOp>(inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<Log2fOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<Log2fOp>(aNaN));

  EXPECT_SIMD_EQ(splat(neg_inf), wrap_vector<Log2fOp>(0.0f));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<Log2fOp>(1.0f));
}

TEST_F(LlvmLibcVecLog2fTest, TrickyInputs) {
  constexpr int N = 10;
  constexpr uint32_t INPUTS[N] = {
      0x3f7d'57f5U, 0x3f7e'3274U, 0x3f7e'd848U, 0x3f7f'd6ccU, 0x3f7f'ffffU,
      0x3f80'079bU, 0x3f81'd0b5U, 0x3f82'e602U, 0x3f83'c98dU, 0x3f8c'ba39U,
  };

  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_SIMD_EQ(wrap_ref<Log2fOp>(x, -x), wrap_vector<Log2fOp>(x, -x));
  }
}

TEST_F(LlvmLibcVecLog2fTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(Log2fOp);
}
