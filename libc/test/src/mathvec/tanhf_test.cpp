//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD tanh.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/tanhf.h"
#include "src/mathvec/tanhf.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecTanhfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using TanhfOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::tanhf,
                                              LIBC_NAMESPACE::tanhf>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecTanhfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<TanhfOp>(aNaN));

  EXPECT_SIMD_EQ(splat(1.0f), wrap_vector<TanhfOp>(inf));

  EXPECT_SIMD_EQ(splat(-1.0f), wrap_vector<TanhfOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<TanhfOp>(0.0f));
}

TEST_F(LlvmLibcVecTanhfTest, ExceptionalValues) {
  constexpr int N = 4;
  constexpr uint32_t INPUTS[N] = {
      0x0040'0000,
      0x1780'0000,
      0x3a12'85ff,
      0x4058'e0a3,
  };

  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_SIMD_EQ(wrap_ref<TanhfOp>(x, -x), wrap_vector<TanhfOp>(x, -x));
  }
}

TEST_F(LlvmLibcVecTanhfTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(TanhfOp);
}
