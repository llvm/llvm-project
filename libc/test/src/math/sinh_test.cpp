//===-- Unittests for sinh ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/sinh.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcSinhTest = LIBC_NAMESPACE::testing::FPTest<double>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcSinhTest, InDoubleRange) {
  // sinh is odd; test [2^-26, 710) and negate for negatives.
  // Values >= 710 overflow to +/-inf.
  constexpr uint64_t COUNT = 123'451;
  uint64_t START = FPBits(0x1.0p-26).uintval();
  uint64_t STOP = FPBits(710.0).uintval();
  uint64_t STEP = (STOP - START) / COUNT;

  auto test = [&](mpfr::RoundingMode rounding_mode) {
    mpfr::ForceRoundingMode __r(rounding_mode);
    if (!__r.success)
      return;

    uint64_t fails = 0;
    uint64_t count = 0;
    uint64_t cc = 0;
    double mx = 0.0, mr = 0.0;
    double tol = 0.5;

    for (uint64_t i = 0, v = START; i <= COUNT; ++i, v += STEP) {
      double x = FPBits(v).get_val();
      if (FPBits(v).is_inf_or_nan())
        continue;

      // Test positive x.
      double result = LIBC_NAMESPACE::sinh(x);
      ++cc;
      if (!FPBits(result).is_inf_or_nan())
        ++count;
      if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Sinh, x, result,
                                             0.5, rounding_mode)) {
        ++fails;
        while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Sinh, x,
                                                  result, tol, rounding_mode)) {
          mx = x;
          mr = result;
          if (tol > 1000.0)
            break;
          tol *= 2.0;
        }
      }

      // Test negative x (sinh is odd).
      double neg_x = -x;
      double neg_result = LIBC_NAMESPACE::sinh(neg_x);
      ++cc;
      if (!FPBits(neg_result).is_inf_or_nan())
        ++count;
      if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Sinh, neg_x,
                                             neg_result, 0.5, rounding_mode)) {
        ++fails;
        while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(
            mpfr::Operation::Sinh, neg_x, neg_result, tol, rounding_mode)) {
          mx = neg_x;
          mr = neg_result;
          if (tol > 1000.0)
            break;
          tol *= 2.0;
        }
      }
    }
    if (fails) {
      tlog << " Sinh failed: " << fails << "/" << count << "/" << cc
           << " tests.\n";
      tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
      EXPECT_MPFR_MATCH(mpfr::Operation::Sinh, mx, mr, 0.5, rounding_mode);
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
