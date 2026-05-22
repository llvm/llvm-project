//===-- Unittests for exp10f ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/exp10f.h"
#include "src/mathvec/exp10f.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecExp10fTest = LIBC_NAMESPACE::testing::FPTest<float>;

using Exp10fOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::exp10f,
                                              LIBC_NAMESPACE::exp10f>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecExp10fTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<Exp10fOp>(aNaN));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Exp10fOp>(inf));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<Exp10fOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(1.0f), wrap_vector<Exp10fOp>(0.0f));

  EXPECT_SIMD_EQ(splat(1.0f), wrap_vector<Exp10fOp>(-0.0f));
}

TEST_F(LlvmLibcVecExp10fTest, Overflow) {
  float x = FPBits(0x7f7fffffU).get_val();
  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Exp10fOp>(x));

  x = FPBits(0x421a209aU).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Exp10fOp>(x), wrap_vector<Exp10fOp>(x));

  x = FPBits(0x421a209bU).get_val();
  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Exp10fOp>(x));

  x = FPBits(0x421a209cU).get_val();
  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Exp10fOp>(x));
}

TEST_F(LlvmLibcVecExp10fTest, Underflow) {
  float x = FPBits(0xff7fffffU).get_val();
  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<Exp10fOp>(x));

  // Values around log10(2^-150).
  x = FPBits(0xc2349e34U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Exp10fOp>(x, 1.0), wrap_vector<Exp10fOp>(x, 1.0));

  x = FPBits(0xc2349e35U).get_val();
  EXPECT_SIMD_EQ(wrap_ref<Exp10fOp>(x, 1.0), wrap_vector<Exp10fOp>(x, 1.0));

  x = FPBits(0xc2349e36U).get_val();
  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<Exp10fOp>(x));
}

TEST_F(LlvmLibcVecExp10fTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = 0x7f80'0000U / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    TEST_VARIED_CASES(x, Exp10fOp);
  }
}
