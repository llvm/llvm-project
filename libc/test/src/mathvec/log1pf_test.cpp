//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for single-precision SIMD log1p.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/log1pf.h"
#include "src/mathvec/log1pf.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecLog1pfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using Log1pfOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, LIBC_NAMESPACE::log1pf,
                                              LIBC_NAMESPACE::log1pf>;
using LIBC_NAMESPACE::cpp::splat;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;

TEST_F(LlvmLibcVecLog1pfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(splat(inf), wrap_vector<Log1pfOp>(inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<Log1pfOp>(neg_inf));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<Log1pfOp>(aNaN));

  EXPECT_SIMD_EQ(splat(0.0f), wrap_vector<Log1pfOp>(0.0f));

  EXPECT_SIMD_EQ(splat(neg_inf), wrap_vector<Log1pfOp>(-1.0f));

  EXPECT_SIMD_EQ(splat(aNaN), wrap_vector<Log1pfOp>(-2.0f));
}

TEST_F(LlvmLibcVecLog1pfTest, TrickyInputs) {
  constexpr int N = 28;
  constexpr uint32_t INPUTS[N] = {
      0x35c00006U, /*0x1.80000cp-20f*/
      0x35400003U, /*0x1.800006p-21f*/
      0x3640000cU, /*0x1.800018p-19f*/
      0x36c00018U, /*0x1.80003p-18f*/
      0x3710001bU, /*0x1.200036p-17f*/
      0x37400030U, /*0x1.80006p-17f*/
      0x3770004bU, /*0x1.e00096p-17f*/
      0x3b9315c8U, /*0x1.262b9p-8f*/
      0x3c6eb7afU, /*0x1.dd6f5ep-7f*/
      0x3ddbfec3U, /*0x1.b7fd86p-4f*/
      0x3efd81adU, /*0x1.fb035ap-2f*/
      0x41078febU, /*0x1.0f1fd6p+3f*/
      0x4cc1c80bU, /*0x1.839016p+26f*/
      0x55185f82U, /*0x1.30bf04p+43f*/
      0x5cd69e88U, /*0x1.ad3d1p+58f*/
      0x5ee8984eU, /*0x1.d1309cp+62f*/
      0x65d890d3U, /*0x1.b121a6p+76f*/
      0x665e7ca6U, /*0x1.bcf94cp+77f*/
      0x6f31a8ecU, /*0x1.6351d8p+95f*/
      0x79e7ec37U, /*0x1.cfd86ep+116f*/
      0x7a17f30aU, /*0x1.2fe614p+117f*/
      0xb53ffffdU, /*-0x1.7ffffap-21f*/
      0xb70fffe5U, /*-0x1.1fffcap-17f*/
      0xbb0ec8c4U, /*-0x1.1d9188p-9f*/
      0xbc4d092cU, /*-0x1.9a1258p-7f*/
      0xbc657728U, /*-0x1.caee5p-7f*/
      0xbd1d20afU, /*-0x1.3a415ep-5f*/
      0xbf800000U, /*-1.0f*/
  };

  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_SIMD_EQ(wrap_ref<Log1pfOp>(x, -x), wrap_vector<Log1pfOp>(x, -x));
  }
}

TEST_F(LlvmLibcVecLog1pfTest, InFloatRange) {
  TEST_MATHVEC_FLOAT_RANGE(Log1pfOp);
}
