//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD sqrt.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/sqrtf.h"
#include "src/mathvec/sqrtf.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecSqrtfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using SqrtfOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::sqrtf,
                                              LIBC_NAMESPACE::sqrtf>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecSqrtfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<SqrtfOp>(aNaN));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<SqrtfOp>(inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<SqrtfOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<SqrtfOp>(0.0f));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<SqrtfOp>(-1.0f));
}

TEST_F(LlvmLibcVecSqrtfTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(SqrtfOp);
}
