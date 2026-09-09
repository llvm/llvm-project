//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD exp10m1.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/exp10m1f.h"
#include "src/mathvec/exp10m1f.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecExp10m1fTest = LIBC_NAMESPACE::testing::FPTest<float>;

using Exp10m1fOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::exp10m1f,
                                              LIBC_NAMESPACE::exp10m1f>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecExp10m1fTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<Exp10m1fOp>(aNaN));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Exp10m1fOp>(inf));

  EXPECT_SIMD_EQ(splat(-1.0f), wrap_vector<Exp10m1fOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<Exp10m1fOp>(0.0f));

  EXPECT_SIMD_EQ(splat(-0.0f), wrap_vector<Exp10m1fOp>(-0.0f));
}

TEST_F(LlvmLibcVecExp10m1fTest, TrickyInputs) {
  constexpr LIBC_NAMESPACE::cpp::array<float, 39> INPUTS = {
      // EXP10M1F_EXCEPTS_LO
      0x1.0fe54ep-11f,
      0x1.80e6eap-11f,
      -0x1.2a33bcp-51f,
      -0x0p+0f,
      -0x1.b59e08p-31f,
      -0x1.bf342p-12f,
      -0x1.6207fp-11f,
      -0x1.bd0c66p-11f,
      -0x1.ffd84cp-10f,
      -0x1.a74172p-9f,
      -0x1.cb694cp-9f,
      // EXP10M1F_EXCEPTS_HI
      0x1.8d31eep-8f,
      0x1.915fcep-8f,
      0x1.bcf982p-8f,
      0x1.99ff0ap-7f,
      0x1.75ea14p-6f,
      0x1.f81b64p-6f,
      0x1.fafecp+3f,
      -0x1.3bf094p-8f,
      -0x1.4558bcp-8f,
      -0x1.4bb43p-8f,
      -0x1.776cc8p-8f,
      -0x1.f024cp-8f,
      -0x1.f510eep-8f,
      -0x1.0b43c4p-7f,
      -0x1.245ee4p-7f,
      -0x1.f9f2dap-7f,
      -0x1.08e42p-6f,
      -0x1.0cdc44p-5f,
      -0x1.ca4322p-5f,
      // Exceptional integers.
      8.0f,
      9.0f,
      10.0f,
      // Overflow boundaries.
      0x1.344134p+5f,
      0x1.344136p+5f,
      0x1.344138p+5f,
      // Underflow boundaries.
      -0x1.e1a5e0p+2f,
      -0x1.e1a5e2p+2f,
      -0x1.e1a5e4p+2f,
  };

  for (float x : INPUTS) {
    EXPECT_SIMD_EQ(wrap_ref<Exp10m1fOp>(x), wrap_vector<Exp10m1fOp>(x));
    EXPECT_SIMD_EQ(wrap_ref<Exp10m1fOp>(x, inf),
                   wrap_vector<Exp10m1fOp>(x, inf));
    EXPECT_SIMD_EQ(wrap_ref<Exp10m1fOp>(x, 0.0),
                   wrap_vector<Exp10m1fOp>(x, 0.0));
  }
}

TEST_F(LlvmLibcVecExp10m1fTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(Exp10m1fOp);
}
