//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD cosh.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/coshf.h"
#include "src/mathvec/coshf.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecCoshfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using CoshfOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::coshf,
                                              LIBC_NAMESPACE::coshf>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecCoshfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<CoshfOp>(aNaN));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<CoshfOp>(inf));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<CoshfOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(1.0f), wrap_vector<CoshfOp>(0.0f));
}

TEST_F(LlvmLibcVecCoshfTest, Overflow) {
  float x = FPBits(0x7f7fffffU).get_val();
  EXPECT_SIMD_EQ(wrap_ref<CoshfOp>(x, -x), wrap_vector<CoshfOp>(x, -x));

  x = FPBits(0x42cffff8U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<CoshfOp>(x, -x), wrap_vector<CoshfOp>(x, -x));

  x = FPBits(0x42d00008U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<CoshfOp>(x, -x), wrap_vector<CoshfOp>(x, -x));
}

TEST_F(LlvmLibcVecCoshfTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(CoshfOp);
}
