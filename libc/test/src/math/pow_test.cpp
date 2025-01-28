//===-- Unittests for pow -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/pow.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcPowTest = LIBC_NAMESPACE::testing::FPTest<double>;
using LIBC_NAMESPACE::testing::tlog;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcPowTest, TrickyInputs) {
  constexpr mpfr::BinaryInput<double> INPUTS[] = {
      {0x1.0853408534085p-2, 0x1.0d148e03bcba8p-1},
      {0x1.65fbd65fbd657p-1, 0x1.f10d148e03bb6p+1},
      {0x1.c046a084d2e12p-1, 0x1.1f9p+12},
      {0x1.ae37ed1670326p-1, 0x1.f967df66a202p-1},
      {0x1.ffffffffffffcp-1, 0x1.fffffffffffffp-2},
      {0x1.f558a88a8aadep-1, 0x1.88ap+12},
      {0x1.e84d32731e593p-1, 0x1.2cb8p+13},
      {0x1.ffffffffffffcp-1, 0x1.fffffffffffffp-2},
  };

  for (auto input : INPUTS) {
    double x = input.x;
    double y = input.y;
    EXPECT_MPFR_MATCH(mpfr::Operation::Pow, input, LIBC_NAMESPACE::pow(x, y),
                      1.5);
  }
}

TEST_F(LlvmLibcPowTest, InFloatRange) {
  constexpr uint64_t X_COUNT = 123;
  constexpr uint64_t X_START = FPBits(0.25).uintval();
  constexpr uint64_t X_STOP = FPBits(4.0).uintval();
  constexpr uint64_t X_STEP = (X_STOP - X_START) / X_COUNT;

  constexpr uint64_t Y_COUNT = 137;
  constexpr uint64_t Y_START = FPBits(0.25).uintval();
  constexpr uint64_t Y_STOP = FPBits(4.0).uintval();
  constexpr uint64_t Y_STEP = (Y_STOP - Y_START) / Y_COUNT;

  auto test = [&](mpfr::RoundingMode rounding_mode) {
    mpfr::ForceRoundingMode __r(rounding_mode);
    if (!__r.success)
      return;

    uint64_t fails = 0;
    uint64_t count = 0;
    uint64_t cc = 0;
    double mx = 0.0, my = 0.0, mr = 0.0;
    double tol = 1.5;

    for (uint64_t i = 0, v = X_START; i <= X_COUNT; ++i, v += X_STEP) {
      double x = FPBits(v).get_val();
      if (FPBits(x).is_inf_or_nan() || x < 0.0)
        continue;

      for (uint64_t j = 0, w = Y_START; j <= Y_COUNT; ++j, w += Y_STEP) {
        double y = FPBits(w).get_val();
        if (FPBits(y).is_inf_or_nan())
          continue;

        double result = LIBC_NAMESPACE::pow(x, y);
        ++cc;
        if (FPBits(result).is_inf_or_nan())
          continue;

        ++count;
        mpfr::BinaryInput<double> inputs{x, y};

        if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Pow, inputs,
                                               result, 1.5, rounding_mode)) {
          ++fails;
          while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(
              mpfr::Operation::Pow, inputs, result, tol, rounding_mode)) {
            mx = x;
            my = y;
            mr = result;

            if (tol > 1000.0)
              break;

            tol *= 2.0;
          }
        }
      }
    }
    if (fails || (count < cc)) {
      tlog << " Pow failed: " << fails << "/" << count << "/" << cc
           << " tests.\n"
           << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
    }
    if (fails) {
      mpfr::BinaryInput<double> inputs{mx, my};
      EXPECT_MPFR_MATCH(mpfr::Operation::Pow, inputs, mr, 1.5, rounding_mode);
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
