//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD cbrt.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/cbrtf.h"
#include "src/mathvec/cbrtf.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecCbrtfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using CbrtfOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::cbrtf,
                                              LIBC_NAMESPACE::cbrtf>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecCbrtfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<CbrtfOp>(aNaN));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<CbrtfOp>(inf));

  EXPECT_SIMD_EQ(splat(neg_inf), wrap_vector<CbrtfOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<CbrtfOp>(0.0f));
}

TEST_F(LlvmLibcVecCbrtfTest, SpecialValues) {
  constexpr float INPUTS[] = {
      0x1.60451p2f, 0x1.31304cp1f, 0x1.d17cp2f, 0x1.bp-143f, 0x1.338cp2f,
  };

  for (float x : INPUTS)
    EXPECT_SIMD_EQ(wrap_ref<CbrtfOp>(x, -x), wrap_vector<CbrtfOp>(x, -x));
}

TEST_F(LlvmLibcVecCbrtfTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(CbrtfOp);
}
