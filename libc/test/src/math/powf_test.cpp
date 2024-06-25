//===-- Unittests for powf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/powf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcPowfTest = LIBC_NAMESPACE::testing::FPTest<float>;
using LIBC_NAMESPACE::testing::tlog;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcPowfTest, TrickyInputs) {
  constexpr int N = 13;
  constexpr mpfr::BinaryInput<float> INPUTS[N] = {
      {0x1.290bbp-124f, 0x1.1e6d92p-25f},
      {0x1.2e9fb6p+5f, -0x1.1b82b6p-18f},
      {0x1.6877f6p+60f, -0x1.75f1c6p-4f},
      {0x1.0936acp-63f, -0x1.55200ep-15f},
      {0x1.d6d72ap+43f, -0x1.749ccap-5f},
      {0x1.4afb2ap-40f, 0x1.063198p+0f},
      {0x1.0124dep+0f, -0x1.fdb016p+9f},
      {0x1.1058p+0f, 0x1.ap+64f},
      {0x1.1058p+0f, -0x1.ap+64f},
      {0x1.1058p+0f, 0x1.ap+64f},
      {0x1.fa32d4p-1f, 0x1.67a62ep+12f},
      {-0x1.8p-49, 0x1.8p+1},
      {0x1.8p-48, 0x1.8p+1},
  };

  for (int i = 0; i < N; ++i) {
    float x = INPUTS[i].x;
    float y = INPUTS[i].y;
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Pow, INPUTS[i],
                                   LIBC_NAMESPACE::powf(x, y), 0.5);
  }
}

TEST_F(LlvmLibcPowfTest, InFloatRange) {
  constexpr uint32_t X_COUNT = 1'23;
  constexpr uint32_t X_START = FPBits(0.25f).uintval();
  constexpr uint32_t X_STOP = FPBits(4.0f).uintval();
  constexpr uint32_t X_STEP = (X_STOP - X_START) / X_COUNT;

  constexpr uint32_t Y_COUNT = 1'37;
  constexpr uint32_t Y_START = FPBits(0.25f).uintval();
  constexpr uint32_t Y_STOP = FPBits(4.0f).uintval();
  constexpr uint32_t Y_STEP = (Y_STOP - Y_START) / Y_COUNT;

  auto test = [&](mpfr::RoundingMode rounding_mode) {
    mpfr::ForceRoundingMode __r(rounding_mode);
    if (!__r.success)
      return;

    uint64_t fails = 0;
    uint64_t count = 0;
    uint64_t cc = 0;
    float mx, my, mr = 0.0;
    double tol = 0.5;

    for (uint32_t i = 0, v = X_START; i <= X_COUNT; ++i, v += X_STEP) {
      float x = FPBits(v).get_val();
      if (isnan(x) || isinf(x) || x < 0.0)
        continue;

      for (uint32_t j = 0, w = Y_START; j <= Y_COUNT; ++j, w += Y_STEP) {
        float y = FPBits(w).get_val();
        if (isnan(y) || isinf(y))
          continue;

        LIBC_NAMESPACE::libc_errno = 0;
        float result = LIBC_NAMESPACE::powf(x, y);
        ++cc;
        if (isnan(result) || isinf(result))
          continue;

        ++count;
        mpfr::BinaryInput<float> inputs{x, y};

        if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Pow, inputs,
                                               result, 0.5, rounding_mode)) {
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
      tlog << " Powf failed: " << fails << "/" << count << "/" << cc
           << " tests.\n"
           << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
    }
    if (fails) {
      mpfr::BinaryInput<float> inputs{mx, my};
      EXPECT_MPFR_MATCH(mpfr::Operation::Pow, inputs, mr, 0.5, rounding_mode);
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
