//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD acosh.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/acoshf.h"
#include "src/mathvec/acoshf.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecAcoshfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using AcoshfOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::acoshf,
                                              LIBC_NAMESPACE::acoshf>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecAcoshfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcoshfOp>(aNaN));

  EXPECT_SIMD_EQ(splat(inf), wrap_vector<AcoshfOp>(inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcoshfOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<AcoshfOp>(0.0f));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<AcoshfOp>(1.0f));
}

TEST_F(LlvmLibcVecAcoshfTest, SpecificBitPatterns) {
  constexpr int N = 17;
  constexpr uint32_t INPUTS[N] = {
      0x3f800000, // x = 1.0f
      0x45abaf26, // x = 0x1.575e4cp12f
      0x45dc6414, // x = 0x1.b8c828p12f
      0x49d29048, // x = 0x1.a5209p20f
      0x4bdd65a5, // x = 0x1.bacb4ap24f
      0x4c803f2c, // x = 0x1.007e58p26f
      0x4f8ffb03, // x = 0x1.1ff606p32f
      0x5c569e88, // x = 0x1.ad3d1p57f
      0x5e68984e, // x = 0x1.d1309cp61f
      0x62f7a05a, // x = 0x1.ef40b4p70f
      0x655890d3, // x = 0x1.b121a6p75f
      0x65de7ca6, // x = 0x1.bcf94cp76f
      0x6eb1a8ec, // x = 0x1.6351d8p94f
      0x71699003, // x = 0x1.d32006p99f
      0x76be09de, // x = 0x1.7c13bcp110f
      0x7967ec37, // x = 0x1.cfd86ep115f
      0x7997f30a, // x = 0x1.2fe614p116f
  };

  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_SIMD_EQ(wrap_ref<AcoshfOp>(x, -x), wrap_vector<AcoshfOp>(x, -x));
  }
}

TEST_F(LlvmLibcVecAcoshfTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(AcoshfOp);
}
