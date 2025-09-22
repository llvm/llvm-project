//===-- Unittests for atan2f ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/atan2f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcAtan2fTest = LIBC_NAMESPACE::testing::FPTest<float>;
using LIBC_NAMESPACE::testing::tlog;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcAtan2fTest, TrickyInputs) {
  constexpr int N = 17;
  mpfr::BinaryInput<float> INPUTS[N] = {
      {0x1.0cb3a4p+20f, 0x1.4ebacp+22f},   {0x1.12215p+1f, 0x1.4fabfcp+22f},
      {-0x1.13baaep+41f, 0x1.5bd22ep+23f}, {0x1.1ff7dcp+41f, 0x1.aec0a6p+23f},
      {0x1.2bc794p+23f, 0x1.0bc0c6p+23f},  {0x1.2fba3ap+42f, 0x1.f99456p+23f},
      {0x1.5ea1f8p+27f, 0x1.f2a1aep+23f},  {0x1.7a931p+44f, 0x1.352ac4p+22f},
      {0x1.8802bcp+21f, 0x1.8f130ap+23f},  {0x1.658ef8p+17f, 0x1.3c00f4p+22f},
      {0x1.69fb0cp+21f, 0x1.39e4c4p+23f},  {0x1.8eb24cp+11f, 0x1.36518p+23f},
      {0x1.9e7ebp+30f, 0x1.d80522p+23f},   {0x1.b4bdeep+19f, 0x1.c19b4p+23f},
      {0x1.bc201p+43f, 0x1.617346p+23f},   {0x1.c96c3cp+20f, 0x1.c01d1ep+23f},
      {0x1.781fcp+28f, 0x1.dcb3cap+23f},
  };

  for (int i = 0; i < N; ++i) {
    float x = INPUTS[i].x;
    float y = INPUTS[i].y;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan2, INPUTS[i],
                                   LIBC_NAMESPACE::atan2f(x, y), 0.5);
    INPUTS[i].x = -INPUTS[i].x;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan2, INPUTS[i],
                                   LIBC_NAMESPACE::atan2f(-x, y), 0.5);
    INPUTS[i].y = -INPUTS[i].y;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan2, INPUTS[i],
                                   LIBC_NAMESPACE::atan2f(-x, -y), 0.5);
    INPUTS[i].x = -INPUTS[i].x;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan2, INPUTS[i],
                                   LIBC_NAMESPACE::atan2f(x, -y), 0.5);
  }
}

TEST_F(LlvmLibcAtan2fTest, InFloatRange) {
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
    uint64_t finite_count = 0;
    uint64_t total_count = 0;
    float failed_x, failed_y, failed_r = 0.0;
    double tol = 0.5;

    for (uint32_t i = 0, v = X_START; i <= X_COUNT; ++i, v += X_STEP) {
      float x = FPBits(v).get_val();
      if (FPBits(v).is_nan() || FPBits(v).is_inf() || x < 0.0)
        continue;

      for (uint32_t j = 0, w = Y_START; j <= Y_COUNT; ++j, w += Y_STEP) {
        float y = FPBits(w).get_val();
        if (FPBits(w).is_nan() || FPBits(w).is_inf())
          continue;

        libc_errno = 0;
        float result = LIBC_NAMESPACE::atan2f(x, y);
        ++total_count;
        if (FPBits(result).is_nan() || FPBits(result).is_inf())
          continue;

        ++finite_count;
        mpfr::BinaryInput<float> inputs{x, y};

        if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Atan2, inputs,
                                               result, 0.5, rounding_mode)) {
          ++fails;
          while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(
              mpfr::Operation::Atan2, inputs, result, tol, rounding_mode)) {
            failed_x = x;
            failed_y = y;
            failed_r = result;

            if (tol > 1000.0)
              break;

            tol *= 2.0;
          }
        }
      }
    }
    if (fails || (finite_count < total_count)) {
      tlog << " Atan2f failed: " << fails << "/" << finite_count << "/"
           << total_count << " tests.\n"
           << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
    }
    if (fails) {
      mpfr::BinaryInput<float> inputs{failed_x, failed_y};
      EXPECT_MPFR_MATCH(mpfr::Operation::Atan2, inputs, failed_r, 0.5,
                        rounding_mode);
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
