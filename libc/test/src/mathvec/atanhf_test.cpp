//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD atanh.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/atanhf.h"
#include "src/mathvec/atanhf.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecAtanhfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using AtanhfOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::atanhf,
                                              LIBC_NAMESPACE::atanhf>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecAtanhfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AtanhfOp>(aNaN));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<AtanhfOp>(0.0f));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<AtanhfOp>(1.0f));

  EXPECT_SIMD_EQ(splat(neg_inf), wrap_vector<AtanhfOp>(-1.0f));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AtanhfOp>(inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AtanhfOp>(neg_inf));

  float x = FPBits(0x3f80'0001U).get_val();
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AtanhfOp>(x, -x));
}

TEST_F(LlvmLibcVecAtanhfTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(AtanhfOp);
}
