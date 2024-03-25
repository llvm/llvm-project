//===-- Unittests for erff ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/math-macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/erff.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcErffTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcErffTest, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::erff(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::erff(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(-1.0f, LIBC_NAMESPACE::erff(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::erff(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::erff(neg_zero));
}

TEST_F(LlvmLibcErffTest, TrickyInputs) {
  constexpr int N = 2;
  constexpr uint32_t INPUTS[N] = {
      0x3f65'9229U, // |x| = 0x1.cb2452p-1f
      0x4004'1e6aU, // |x| = 0x1.083cd4p+1f
  };
  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Erf, x,
                                   LIBC_NAMESPACE::erff(x), 0.5);
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Erf, -x,
                                   LIBC_NAMESPACE::erff(-x), 0.5);
  }
}

TEST_F(LlvmLibcErffTest, InFloatRange) {
  constexpr uint32_t COUNT = 234561;
  constexpr uint32_t START = 0;           // 0
  constexpr uint32_t STOP = 0x4080'0000U; // 4.0f

  constexpr uint64_t STEP = (STOP - START) / COUNT;

  auto test = [&](mpfr::RoundingMode rounding_mode) {
    mpfr::ForceRoundingMode __r(rounding_mode);
    if (!__r.success)
      return;

    uint32_t fails = 0;
    uint32_t count = 0;
    uint32_t cc = 0;
    float mx, mr = 0.0;
    double tol = 0.5;

    for (uint32_t i = 0, v = START; i <= COUNT; ++i, v += STEP) {
      float x = FPBits(v).get_val();
      if (isnan(x))
        continue;

      float result = LIBC_NAMESPACE::erff(x);
      ++cc;
      if (isnan(result))
        continue;

      ++count;
      if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Erf, x, result,
                                             0.5, rounding_mode)) {
        ++fails;
        while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Erf, x,
                                                  result, tol, rounding_mode)) {
          mx = x;
          mr = result;
          tol *= 2.0;
        }
      }
    }
    tlog << " Log failed: " << fails << "/" << count << "/" << cc
         << " tests.\n";
    tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
    if (fails) {
      EXPECT_MPFR_MATCH(mpfr::Operation::Erf, mx, mr, 0.5, rounding_mode);
    }
  };

  tlog << " Test Rounding To Nearest...\n";
  test(mpfr::RoundingMode::Nearest);

  tlog << " Test Rounding Downward...\n";
  test(mpfr::RoundingMode::Downward);

  tlog << " Test Rounding Upward...\n";
  test(mpfr::RoundingMode::Upward);

  tlog << " Test Rounding Toward Zero...\n";
  test(mpfr::RoundingMode::TowardZero);
}
