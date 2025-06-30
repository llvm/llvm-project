//===-- Unittests for exp2m1f ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/array.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/math/exp2m1f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <stdint.h>

using LlvmLibcExp2m1fTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcExp2m1fTest, TrickyInputs) {
  constexpr LIBC_NAMESPACE::cpp::array<float, 10> INPUTS = {
      // EXP2M1F_EXCEPTS_LO
      0x1.36dc8ep-36,
      0x1.224936p-19,
      0x1.d16d2p-20,
      0x1.17949ep-14,
      -0x1.9c3e1ep-38,
      -0x1.4d89b4p-32,
      -0x1.a6eac4p-10,
      -0x1.e7526ep-6,
      // EXP2M1F_EXCEPTS_HI
      0x1.16a972p-1,
      -0x1.9f12acp-5,
  };

  for (float x : INPUTS) {
    libc_errno = 0;
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2m1, x,
                                   LIBC_NAMESPACE::exp2m1f(x), 0.5);
  }
}

TEST_F(LlvmLibcExp2m1fTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (FPBits(v).is_nan() || FPBits(v).is_inf())
      continue;
    libc_errno = 0;
    float result = LIBC_NAMESPACE::exp2m1f(x);

    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (FPBits(result).is_nan() || FPBits(result).is_inf() || libc_errno != 0)
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2m1, x,
                                   LIBC_NAMESPACE::exp2m1f(x), 0.5);
  }
}
