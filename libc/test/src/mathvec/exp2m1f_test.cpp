//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD exp2m1.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/exp2m1f.h"
#include "src/mathvec/exp2m1f.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecExp2m1fTest = LIBC_NAMESPACE::testing::FPTest<float>;

using Exp2m1fOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::exp2m1f,
                                              LIBC_NAMESPACE::exp2m1f>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecExp2m1fTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<Exp2m1fOp>(aNaN));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Exp2m1fOp>(inf));

  EXPECT_SIMD_EQ(splat(-1.0f), wrap_vector<Exp2m1fOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<Exp2m1fOp>(0.0f));

  EXPECT_SIMD_EQ(splat(-0.0f), wrap_vector<Exp2m1fOp>(-0.0f));
}

TEST_F(LlvmLibcVecExp2m1fTest, TrickyInputs) {
  constexpr LIBC_NAMESPACE::cpp::array<float, 10> INPUTS = {
      // EXP2M1F_EXCEPTS_LO
      0x1.36dc8ep-36,
      0x1.224936p-19,
      0x1.d16d2p-20,
      0x1.17949ep-14,
      -0x1.9c3e1ep-38,
      -0x1.4d89b4p-32,
      -0x1.a6eac4p-10,
      -0x1.e7526ep-6,
      // EXP2M1F_EXCEPTS_HI
      0x1.16a972p-1,
      -0x1.9f12acp-5,
  };

  for (float x : INPUTS) {
    EXPECT_SIMD_EQ(wrap_ref<Exp2m1fOp>(x), wrap_vector<Exp2m1fOp>(x));
    EXPECT_SIMD_EQ(wrap_ref<Exp2m1fOp>(x, inf), wrap_vector<Exp2m1fOp>(x, inf));
    EXPECT_SIMD_EQ(wrap_ref<Exp2m1fOp>(x, 0.0), wrap_vector<Exp2m1fOp>(x, 0.0));
  }
}

TEST_F(LlvmLibcVecExp2m1fTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(Exp2m1fOp);
}
