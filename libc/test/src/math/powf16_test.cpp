//===-- Unittests for powf16 ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "src/math/powf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcPowF16Test = LIBC_NAMESPACE::testing::FPTest<float16>;
using LIBC_NAMESPACE::testing::tlog;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcPowF16Test, TrickyInputs) {
  // These values are in half precision.
  constexpr mpfr::BinaryInput<float16> INPUTS[] = {
      {static_cast<float16>(0x1.08p-2f), static_cast<float16>(0x1.0cp-1f)},
      {static_cast<float16>(0x1.66p-1f), static_cast<float16>(0x1.f1p+1f)},
      {static_cast<float16>(0x1.c04p-1f), static_cast<float16>(0x1.2p+12f)},
      {static_cast<float16>(0x1.aep-1f), static_cast<float16>(0x1.f9p-1f)},
      {static_cast<float16>(0x1.ffcp-1f), static_cast<float16>(0x1.fffp-2f)},
      {static_cast<float16>(0x1.f55p-1f), static_cast<float16>(0x1.88p+12f)},
      {static_cast<float16>(0x1.e84p-1f), static_cast<float16>(0x1.2cp+13f)},
  };

  for (auto input : INPUTS) {
    float16 x = input.x;
    float16 y = input.y;
    EXPECT_MPFR_MATCH(mpfr::Operation::Pow, input, LIBC_NAMESPACE::powf16(x, y),
                      1.0); // 1 ULP tolerance is enough for f16
  }
}

TEST_F(LlvmLibcPowF16Test, InFloat16Range) {
  constexpr uint16_t X_COUNT = 63;
  constexpr uint16_t X_START = FPBits(static_cast<float16>(0.25)).uintval();
  constexpr uint16_t X_STOP = FPBits(static_cast<float16>(4.0)).uintval();
  constexpr uint16_t X_STEP = (X_STOP - X_START) / X_COUNT;

  constexpr uint16_t Y_COUNT = 59;
  constexpr uint16_t Y_START = FPBits(static_cast<float16>(0.25)).uintval();
  constexpr uint16_t Y_STOP = FPBits(static_cast<float16>(4.0)).uintval();
  constexpr uint16_t Y_STEP = (Y_STOP - Y_START) / Y_COUNT;

  auto test = [&](mpfr::RoundingMode rounding_mode) {
    mpfr::ForceRoundingMode __r(rounding_mode);
    if (!__r.success)
      return;

    uint64_t fails = 0;
    uint64_t count = 0;
    uint64_t cc = 0;
    float16 mx = 0.0, my = 0.0, mr = 0.0;
    double tol = 1.0; // start with 1 ULP for half precision

    for (uint16_t i = 0, v = X_START; i <= X_COUNT; ++i, v += X_STEP) {
      float16 x = FPBits(v).get_val();
      if (FPBits(x).is_inf_or_nan() || x < static_cast<float16>(0.0))
        continue;

      for (uint16_t j = 0, w = Y_START; j <= Y_COUNT; ++j, w += Y_STEP) {
        float16 y = FPBits(w).get_val();
        if (FPBits(y).is_inf_or_nan())
          continue;

        float16 result = LIBC_NAMESPACE::powf16(x, y);
        ++cc;
        if (FPBits(result).is_inf_or_nan())
          continue;

        ++count;
        mpfr::BinaryInput<float16> inputs{x, y};

        if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Pow, inputs,
                                               result, 1.0, rounding_mode)) {
          ++fails;
          while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(
              mpfr::Operation::Pow, inputs, result, tol, rounding_mode)) {
            mx = x;
            my = y;
            mr = result;

            if (tol > 128.0) // half precision is only ~11 bits
              break;

            tol *= 2.0;
          }
        }
      }
    }
    if (fails || (count < cc)) {
      tlog << " powf16 failed: " << fails << "/" << count << "/" << cc
           << " tests.\n"
           << "   Max ULPs is at most: " << static_cast<uint64_t>(tol)
           << ".\n";
    }
    if (fails) {
      mpfr::BinaryInput<float16> inputs{mx, my};
      EXPECT_MPFR_MATCH(mpfr::Operation::Pow, inputs, mr, 1.0, rounding_mode);
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


