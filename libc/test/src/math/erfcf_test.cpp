//===-- Unittests for erfcf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/optimization.h"
#include "src/math/erfcf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 1
#else
#define TOLERANCE 0
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

using LlvmLibcErfcfTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcErfcfTest, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::erfcf(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::erfcf(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(2.0f, LIBC_NAMESPACE::erfcf(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::erfcf(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::erfcf(neg_zero));
}

TEST_F(LlvmLibcErfcfTest, TrickyInputs) {
  constexpr int N = 1;
  constexpr uint32_t INPUTS[N] = {
      0x376c9f62U, // |x| = 0x1.d93ec4p-17f
  };
  for (int i = 0; i < N; ++i) {
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Erfc, x,
                                   LIBC_NAMESPACE::erfcf(x), TOLERANCE + 0.5);
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Erfc, -x,
                                   LIBC_NAMESPACE::erfcf(-x), TOLERANCE + 0.5);
  }
}

TEST_F(LlvmLibcErfcfTest, InFloatRange) {
  constexpr uint32_t COUNT = 234561;
  constexpr uint32_t START = 0;           // 0
  constexpr uint32_t STOP = 0x4120'0000U; // 10.0f

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
      if (FPBits(v).is_nan())
        continue;

      float result = LIBC_NAMESPACE::erfcf(x);
      ++cc;
      if (FPBits(result).is_nan())
        continue;

      ++count;
      if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Erfc, x, result,
                                             0.5, rounding_mode)) {
        ++fails;
        while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Erfc, x,
                                                  result, tol, rounding_mode)) {
          mx = x;
          mr = result;
          tol *= 2.0;
        }
      }
    }
    if (fails) {
      tlog << " Log failed: " << fails << "/" << count << "/" << cc
           << " tests.\n";
      tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
      EXPECT_MPFR_MATCH(mpfr::Operation::Erfc, mx, mr, 0.5, rounding_mode);
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
