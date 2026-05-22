//===-- Unittests for rsqrtf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/rsqrtf.h"
#include "src/mathvec/rsqrtf.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecRsqrtfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using RsqrtfOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::rsqrtf,
                                              LIBC_NAMESPACE::rsqrtf>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecRsqrtfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<RsqrtfOp>(aNaN));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<RsqrtfOp>(inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<RsqrtfOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<RsqrtfOp>(0.0f));

  EXPECT_SIMD_EQ(splat(neg_inf), wrap_vector<RsqrtfOp>(-0.0f));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<RsqrtfOp>(-1.0f));
}

TEST_F(LlvmLibcVecRsqrtfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = 0x7f80'0000U / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    TEST_VARIED_CASES(x, RsqrtfOp);
  }
}
