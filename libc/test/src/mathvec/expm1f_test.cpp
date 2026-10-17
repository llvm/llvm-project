//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD expm1.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/expm1f.h"
#include "src/mathvec/expm1f.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecExpm1fTest = LIBC_NAMESPACE::testing::FPTest<float>;

using Expm1fOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::expm1f,
                                              LIBC_NAMESPACE::expm1f>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecExpm1fTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<Expm1fOp>(aNaN));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Expm1fOp>(inf));

  EXPECT_SIMD_EQ(splat(-1.0f), wrap_vector<Expm1fOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<Expm1fOp>(0.0f));

  EXPECT_SIMD_EQ(splat(-0.0f), wrap_vector<Expm1fOp>(-0.0f));
}

TEST_F(LlvmLibcVecExpm1fTest, Overflow) {
  float x = FPBits(0x7f7fffffU).get_val();
  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Expm1fOp>(x));

  x = FPBits(0x42cffff8U).get_val();
  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Expm1fOp>(x));

  x = FPBits(0x42d00008U).get_val();
  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Expm1fOp>(x));
}

TEST_F(LlvmLibcVecExpm1fTest, Underflow) {
  float x = FPBits(0xff7fffffU).get_val();
  EXPECT_SIMD_EQ(splat(-1.0f), wrap_vector<Expm1fOp>(x));

  x = FPBits(0xc2cffff8U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Expm1fOp>(x, 1.0), wrap_vector<Expm1fOp>(x, 1.0));

  x = FPBits(0xc2d00008U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Expm1fOp>(x, 1.0), wrap_vector<Expm1fOp>(x, 1.0));
}

TEST_F(LlvmLibcVecExpm1fTest, Borderline) {
  float x;

  x = FPBits(0x42affff8U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Expm1fOp>(x, 1.0), wrap_vector<Expm1fOp>(x, 1.0));

  x = FPBits(0x42b00008U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Expm1fOp>(x, 1.0), wrap_vector<Expm1fOp>(x, 1.0));

  x = FPBits(0xc2affff8U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Expm1fOp>(x, 1.0), wrap_vector<Expm1fOp>(x, 1.0));

  x = FPBits(0xc2b00008U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Expm1fOp>(x, 1.0), wrap_vector<Expm1fOp>(x, 1.0));

  x = FPBits(0x3dc252ddU).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Expm1fOp>(x, 1.0), wrap_vector<Expm1fOp>(x, 1.0));

  x = FPBits(0x3e35bec5U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Expm1fOp>(x, 1.0), wrap_vector<Expm1fOp>(x, 1.0));

  x = FPBits(0x942ed494U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Expm1fOp>(x, 1.0), wrap_vector<Expm1fOp>(x, 1.0));

  x = FPBits(0xbdc1c6cbU).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Expm1fOp>(x, 1.0), wrap_vector<Expm1fOp>(x, 1.0));
}

TEST_F(LlvmLibcVecExpm1fTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(Expm1fOp);
}
