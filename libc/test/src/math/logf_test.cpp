//===-- Unittests for logf-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/math-macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/logf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <stdint.h>

using LlvmLibcLogfTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcLogfTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::logf(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::logf(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::logf(neg_inf), FE_INVALID);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::logf(0.0f),
                              FE_DIVBYZERO);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::logf(-0.0f),
                              FE_DIVBYZERO);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::logf(-1.0f), FE_INVALID);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::logf(1.0f));
}

TEST_F(LlvmLibcLogfTest, TrickyInputs) {
  constexpr int N = 35;
  constexpr uint32_t INPUTS[N] = {
      0x1b7679ffU, /*0x1.ecf3fep-73f*/
      0x1e88452dU, /*0x1.108a5ap-66f*/
      0x3f800001U, /*0x1.000002p+0f*/
      0x3509dcf6U, /*0x1.13b9ecp-21f*/
      0x3bf86ef0U, /*0x1.f0ddep-8f*/
      0x3c413d3aU, /*0x1.827a74p-7f*/
      0x3ca1c99fU, /*0x1.43933ep-6f*/
      0x3d13e105U, /*0x1.27c20ap-5f*/
      0x3f7ff1f2U, /*0x1.ffe3e4p-1f*/
      0x3f7fffffU, /*0x1.fffffep-1f*/
      0x3f800001U, /*0x1.000002p+0f*/
      0x3f800006U, /*0x1.00000cp+0f*/
      0x3f800014U, /*0x1.000028p+0f*/
      0x3f80001cU, /*0x1.000038p+0f*/
      0x3f80c777U, /*0x1.018eeep+0f*/
      0x3f80ce72U, /*0x1.019ce4p+0f*/
      0x3f80d19fU, /*0x1.01a33ep+0f*/
      0x3f80f7bfU, /*0x1.01ef7ep+0f*/
      0x3f80fcfeU, /*0x1.01f9fcp+0f*/
      0x3f81feb4U, /*0x1.03fd68p+0f*/
      0x3f83d731U, /*0x1.07ae62p+0f*/
      0x3f90cb1dU, /*0x1.21963ap+0f*/
      0x3fc55379U, /*0x1.8aa6f2p+0f*/
      0x3fd364d7U, /*0x1.a6c9aep+0f*/
      0x41178febU, /*0x1.2f1fd6p+3f*/
      0x4c5d65a5U, /*0x1.bacb4ap+25f*/
      0x4e85f412U, /*0x1.0be824p+30f*/
      0x500ffb03U, /*0x1.1ff606p+33f*/
      0x5cd69e88U, /*0x1.ad3d1p+58f*/
      0x5ee8984eU, /*0x1.d1309cp+62f*/
      0x65d890d3U, /*0x1.b121a6p+76f*/
      0x665e7ca6U, /*0x1.bcf94cp+77f*/
      0x6f31a8ecU, /*0x1.6351d8p+95f*/
      0x79e7ec37U, /*0x1.cfd86ep+116f*/
      0x7a17f30aU, /*0x1.2fe614p+117f*/
  };
  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log, x,
                                   LIBC_NAMESPACE::logf(x), 0.5);
  }
}

TEST_F(LlvmLibcLogfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log, x,
                                   LIBC_NAMESPACE::logf(x), 0.5);
  }
}
