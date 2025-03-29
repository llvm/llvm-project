//===-- Unittests for exp10m1f --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/array.h"
#include "src/errno/libc_errno.h"
#include "src/math/exp10m1f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <stdint.h>

using LlvmLibcExp10m1fTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcExp10m1fTest, TrickyInputs) {
  constexpr LIBC_NAMESPACE::cpp::array<float, 39> INPUTS = {
      // EXP10M1F_EXCEPTS_LO
      0x1.0fe54ep-11f,
      0x1.80e6eap-11f,
      -0x1.2a33bcp-51f,
      -0x0p+0f,
      -0x1.b59e08p-31f,
      -0x1.bf342p-12f,
      -0x1.6207fp-11f,
      -0x1.bd0c66p-11f,
      -0x1.ffd84cp-10f,
      -0x1.a74172p-9f,
      -0x1.cb694cp-9f,
      // EXP10M1F_EXCEPTS_HI
      0x1.8d31eep-8f,
      0x1.915fcep-8f,
      0x1.bcf982p-8f,
      0x1.99ff0ap-7f,
      0x1.75ea14p-6f,
      0x1.f81b64p-6f,
      0x1.fafecp+3f,
      -0x1.3bf094p-8f,
      -0x1.4558bcp-8f,
      -0x1.4bb43p-8f,
      -0x1.776cc8p-8f,
      -0x1.f024cp-8f,
      -0x1.f510eep-8f,
      -0x1.0b43c4p-7f,
      -0x1.245ee4p-7f,
      -0x1.f9f2dap-7f,
      -0x1.08e42p-6f,
      -0x1.0cdc44p-5f,
      -0x1.ca4322p-5f,
      // Exceptional integers.
      8.0f,
      9.0f,
      10.0f,
      // Overflow boundaries.
      0x1.344134p+5f,
      0x1.344136p+5f,
      0x1.344138p+5f,
      // Underflow boundaries.
      -0x1.e1a5e0p+2f,
      -0x1.e1a5e2p+2f,
      -0x1.e1a5e4p+2f,
  };

  for (float x : INPUTS) {
    LIBC_NAMESPACE::libc_errno = 0;
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10m1, x,
                                   LIBC_NAMESPACE::exp10m1f(x), 0.5);
  }
}

TEST_F(LlvmLibcExp10m1fTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (isnan(x) || isinf(x))
      continue;
    LIBC_NAMESPACE::libc_errno = 0;
    float result = LIBC_NAMESPACE::exp10m1f(x);

    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || LIBC_NAMESPACE::libc_errno != 0)
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10m1, x,
                                   LIBC_NAMESPACE::exp10m1f(x), 0.5);
  }
}
