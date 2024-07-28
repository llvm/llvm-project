//===-- Unittests for log1pf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/log1pf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcLog1pfTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcLog1pfTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::log1pf(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::log1pf(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log1pf(neg_inf), FE_INVALID);
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::log1pf(0.0f));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::log1pf(-0.0f));
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log1pf(-1.0f),
                              FE_DIVBYZERO);
}

TEST_F(LlvmLibcLog1pfTest, TrickyInputs) {
  constexpr int N = 27;
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
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log1p, x,
                                   LIBC_NAMESPACE::log1pf(x), 0.5);
  }
}

TEST_F(LlvmLibcLog1pfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (FPBits(v).is_nan() || FPBits(v).is_inf())
      continue;
    LIBC_NAMESPACE::libc_errno = 0;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log1p, x,
                                   LIBC_NAMESPACE::log1pf(x), 0.5);
  }
}
