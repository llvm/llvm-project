//===-- Unittests for logf-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/logf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcLogfTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, __llvm_libc::logf(aNaN));
  EXPECT_FP_EQ(inf, __llvm_libc::logf(inf));
  EXPECT_TRUE(FPBits((__llvm_libc::logf(neg_inf))).is_nan());
  EXPECT_FP_EQ(neg_inf, __llvm_libc::logf(0.0f));
  EXPECT_FP_EQ(neg_inf, __llvm_libc::logf(-0.0f));
  EXPECT_TRUE(FPBits(__llvm_libc::logf(-1.0f)).is_nan());
  EXPECT_FP_EQ(zero, __llvm_libc::logf(1.0f));
}

TEST(LlvmLibcLogfTest, TrickyInputs) {
  constexpr int N = 28;
  constexpr uint32_t INPUTS[N] = {
      0x3509dcf6U, /*0x1.13b9ecp-21f*/
      0x3bf86ef0U, /*0x1.f0ddep-8f*/
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
      0x65d890d3U, /*0x1.b121a6p+76f*/
      0x6f31a8ecU, /*0x1.6351d8p+95f*/
      0x7a17f30aU, /*0x1.2fe614p+117f*/
  };
  for (int i = 0; i < N; ++i) {
    float x = float(FPBits(INPUTS[i]));
    EXPECT_MPFR_MATCH(mpfr::Operation::Log, x, __llvm_libc::logf(x), 0.5);
  }
}

TEST(LlvmLibcLogfTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    errno = 0;
    float result = __llvm_libc::logf(x);
    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || errno != 0)
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Log, x, __llvm_libc::logf(x), 0.5);
  }
}
