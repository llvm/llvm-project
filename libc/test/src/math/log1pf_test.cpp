//===-- Unittests for log1pf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/log1pf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibclog1pfTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, __llvm_libc::log1pf(aNaN));
  EXPECT_FP_EQ(inf, __llvm_libc::log1pf(inf));
  EXPECT_TRUE(FPBits((__llvm_libc::log1pf(neg_inf))).is_nan());
  EXPECT_FP_EQ(zero, __llvm_libc::log1pf(0.0f));
  EXPECT_FP_EQ(neg_zero, __llvm_libc::log1pf(-0.0f));
  EXPECT_FP_EQ(neg_inf, __llvm_libc::log1pf(-1.0f));
}

TEST(LlvmLibclog1pfTest, TrickyInputs) {
  constexpr int N = 20;
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
      0x41078febU, /*0x1.0f1fd6p+3f*/
      0x5cd69e88U, /*0x1.ad3d1p+58f*/
      0x65d890d3U, /*0x1.b121a6p+76f*/
      0x6f31a8ecU, /*0x1.6351d8p+95f*/
      0x7a17f30aU, /*0x1.2fe614p+117f*/
      0xb53ffffdU, /*-0x1.7ffffap-21f*/
      0xb70fffe5U, /*-0x1.1fffcap-17f*/
      0xbb0ec8c4U, /*-0x1.1d9188p-9f*/
      0xbc4d092cU, /*-0x1.9a1258p-7f*/
      0xbc657728U, /*-0x1.caee5p-7f*/
      0xbd1d20afU, /*-0x1.3a415ep-5f*/
  };
  for (int i = 0; i < N; ++i) {
    float x = float(FPBits(INPUTS[i]));
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log1p, x,
                                   __llvm_libc::log1pf(x), 0.5);
  }
}

TEST(LlvmLibclog1pfTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    errno = 0;
    float result = __llvm_libc::log1pf(x);
    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || errno != 0)
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log1p, x,
                                   __llvm_libc::log1pf(x), 0.5);
  }
}
