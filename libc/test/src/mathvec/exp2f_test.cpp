//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD exp2.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/exp2f.h"
#include "src/mathvec/exp2f.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecExp2fTest = LIBC_NAMESPACE::testing::FPTest<float>;

using Exp2fOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::exp2f,
                                              LIBC_NAMESPACE::exp2f>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecExp2fTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<Exp2fOp>(aNaN));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Exp2fOp>(inf));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<Exp2fOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(1.0f), wrap_vector<Exp2fOp>(0.0f));

  EXPECT_SIMD_EQ(splat(1.0f), wrap_vector<Exp2fOp>(-0.0f));
}

TEST_F(LlvmLibcVecExp2fTest, Overflow) {
  float x = FPBits(0x7f7fffffU).get_val();
  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Exp2fOp>(x));

  x = FPBits(0x43000000U).get_val();
  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Exp2fOp>(x));

  x = FPBits(0x43000001U).get_val();
  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Exp2fOp>(x));
}

TEST_F(LlvmLibcVecExp2fTest, Underflow) {
  float x = FPBits(0xff7fffffU).get_val();
  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<Exp2fOp>(x));

  x = FPBits(0xc3158000U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Exp2fOp>(x, 1.0), wrap_vector<Exp2fOp>(x, 1.0));

  x = FPBits(0xc3160000U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Exp2fOp>(x, 1.0), wrap_vector<Exp2fOp>(x, 1.0));

  x = FPBits(0xc3165432U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Exp2fOp>(x, 1.0), wrap_vector<Exp2fOp>(x, 1.0));
}

TEST_F(LlvmLibcVecExp2fTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(Exp2fOp);
}
