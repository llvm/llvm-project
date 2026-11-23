//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD exp.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/math/expf_double_eval.h"
#include "src/mathvec/expf.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecExpfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using ExpfOp = LIBC_NAMESPACE::testing::mathvec::UnaryOp<
    float, LIBC_NAMESPACE::math::double_eval::expf, LIBC_NAMESPACE::expf>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecExpfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<ExpfOp>(aNaN));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<ExpfOp>(inf));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<ExpfOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(1.0f), wrap_vector<ExpfOp>(0.0f));

  EXPECT_SIMD_EQ(splat(1.0f), wrap_vector<ExpfOp>(-0.0f));
}

TEST_F(LlvmLibcVecExpfTest, Overflow) {
  EXPECT_SIMD_EQ(splat(inf),
                 wrap_vector<ExpfOp>(FPBits(0x7f7fffffU).get_val()));

  EXPECT_SIMD_EQ(splat(inf),
                 wrap_vector<ExpfOp>(FPBits(0x42cffff8U).get_val()));

  EXPECT_SIMD_EQ(splat(inf),
                 wrap_vector<ExpfOp>(FPBits(0x42d00008U).get_val()));
}

TEST_F(LlvmLibcVecExpfTest, Underflow) {
  float x = FPBits(0xff7fffffU).get_val();
  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<ExpfOp>(x));

  x = FPBits(0xc2cffff8U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<ExpfOp>(x, 1.0), wrap_vector<ExpfOp>(x, 1.0));

  x = FPBits(0xc2d00008U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<ExpfOp>(x, 1.0), wrap_vector<ExpfOp>(x, 1.0));
}

TEST_F(LlvmLibcVecExpfTest, Borderline) {
  float x;

  x = FPBits(0x42affff8U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<ExpfOp>(x, 1.0), wrap_vector<ExpfOp>(x, 1.0));

  x = FPBits(0x42b00008U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<ExpfOp>(x, 1.0), wrap_vector<ExpfOp>(x, 1.0));

  x = FPBits(0xc2affff8U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<ExpfOp>(x, 1.0), wrap_vector<ExpfOp>(x, 1.0));

  x = FPBits(0xc2b00008U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<ExpfOp>(x, 1.0), wrap_vector<ExpfOp>(x, 1.0));

  x = FPBits(0xc236bd8cU).get_val();
  EXPECT_SIMD_EQ(wrap_ref<ExpfOp>(x, 1.0), wrap_vector<ExpfOp>(x, 1.0));
}

TEST_F(LlvmLibcVecExpfTest, NonFMAHardToRound) {
  // The middle input needs an explicit correction without double-precision FMA.
  constexpr uint32_t INPUTS[] = {0xc16912ccU, 0xc16912cdU, 0xc16912ceU};
  constexpr size_t LANES =
      LIBC_NAMESPACE::cpp::internal::native_vector_size<float>;
  for (uint32_t bits : INPUTS) {
    float x = FPBits(bits).get_val();
    EXPECT_SIMD_EQ(wrap_ref<ExpfOp>(x), wrap_vector<ExpfOp>(x));
    for (size_t lane = 0; lane < LANES; ++lane) {
      LIBC_NAMESPACE::cpp::simd<float> input(1.0f);
      input[lane] = x;
      EXPECT_SIMD_EQ(ExpfOp::eval_scalar(input), ExpfOp::eval_vector(input));
    }
  }
}

TEST_F(LlvmLibcVecExpfTest, InFloatRange) { TEST_MATHVEC_FLOAT_RANGE(ExpfOp); }
