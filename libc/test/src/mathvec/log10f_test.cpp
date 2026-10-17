//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD log10.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/log10f.h"
#include "src/mathvec/log10f.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecLog10fTest = LIBC_NAMESPACE::testing::FPTest<float>;

using Log10fOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::log10f,
                                              LIBC_NAMESPACE::log10f>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecLog10fTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Log10fOp>(inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<Log10fOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<Log10fOp>(aNaN));

  EXPECT_SIMD_EQ(splat(neg_inf), wrap_vector<Log10fOp>(0.0f));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<Log10fOp>(1.0f));
}

TEST_F(LlvmLibcVecLog10fTest, TrickyInputs) {
  constexpr int N = 21;
  constexpr uint32_t INPUTS[N] = {
      0x3f800000U /*1.0f*/,
      0x41200000U /*10.0f*/,
      0x42c80000U /*100.0f*/,
      0x447a0000U /*1,000.0f*/,
      0x461c4000U /*10,000.0f*/,
      0x47c35000U /*100,000.0f*/,
      0x49742400U /*1,000,000.0f*/,
      0x4b189680U /*10,000,000.0f*/,
      0x4cbebc20U /*100,000,000.0f*/,
      0x4e6e6b28U /*1,000,000,000.0f*/,
      0x501502f9U /*10,000,000,000.0f*/,
      0x4f134f83U /*2471461632.0f*/,
      0x7956ba5eU /*69683218960000541503257137270226944.0f*/,
      0x08ae'a356U /*0x1.5d46acp-110f*/,
      0x1c7d'a337U /*0x1.fb466ep-71f*/,
      0x69c8'c583U /*0x1.918b06p+84f*/,
      0x0efe'ee7aU /*0x1.fddcf4p-98f*/,
      0x3f5f'de1bU /*0x1.bfbc36p-1f*/,
      0x3f80'70d8U /*0x1.00e1bp0f*/,
      0x120b'93dcU /*0x1.1727b8p-91f*/,
      0x13ae'78d3U /*0x1.5cf1a6p-88f*/,
  };

  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_SIMD_EQ(wrap_ref<Log10fOp>(x, -x), wrap_vector<Log10fOp>(x, -x));
  }
}

TEST_F(LlvmLibcVecLog10fTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(Log10fOp);
}
