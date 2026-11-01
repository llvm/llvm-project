//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD acospi.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/acospif.h"
#include "src/mathvec/acospif.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecAcospifTest = LIBC_NAMESPACE::testing::FPTest<float>;

using AcospifOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::acospif,
                                              LIBC_NAMESPACE::acospif>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecAcospifTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcospifOp>(aNaN));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcospifOp>(inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcospifOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcospifOp>(2.0f));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcospifOp>(-2.0f));

  EXPECT_SIMD_EQ(splat(0.5f), wrap_vector<AcospifOp>(0.0f));

  EXPECT_SIMD_EQ(splat(0.5f), wrap_vector<AcospifOp>(-0.0f));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<AcospifOp>(1.0f));

  EXPECT_SIMD_EQ(splat(1.0f), wrap_vector<AcospifOp>(-1.0f));
}

TEST_F(LlvmLibcVecAcospifTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(AcospifOp);
}
