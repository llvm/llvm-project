//===-- Unittests for atan2f16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/optimization.h"
#include "src/math/atan2f16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 1
#else
#define TOLERANCE 0
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

using LlvmLibcAtan2f16Test = LIBC_NAMESPACE::testing::FPTest<float16>;
using LIBC_NAMESPACE::testing::tlog;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcAtan2f16Test, TrickyInputs) {
  constexpr int N = 17;
  mpfr::BinaryInput<float16> INPUTS[N] = {
      {0x1.0p0f, 0x1.4p0f},   {0x1.2p0f, 0x1.0p0f},   {-0x1.0p0f, 0x1.2p0f},
      {0x1.8p0f, 0x1.0p0f},   {0x1.2p-2f, 0x1.0p-2f}, {0x1.0p0f, 0x1.0p0f},
      {0x1.0p-1f, 0x1.0p0f},  {0x1.0p0f, 0x1.0p-1f},  {0x1.8p1f, 0x1.0p0f},
      {0x1.0p0f, 0x1.4p1f},   {0x1.2p1f, 0x1.0p-1f},  {0x1.0p-2f, 0x1.0p0f},
      {0x1.0p0f, 0x1.0p-3f},  {0x1.4p0f, 0x1.0p0f},   {0x1.0p0f, 0x1.8p0f},
      {0x1.0p-1f, 0x1.0p-1f}, {0x1.2p0f, 0x1.2p0f},
  };

  for (int i = 0; i < N; ++i) {
    float16 x = INPUTS[i].x;
    float16 y = INPUTS[i].y;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan2, INPUTS[i],
                                   LIBC_NAMESPACE::atan2f16(x, y),
                                   TOLERANCE + 0.5);
    INPUTS[i].x = -INPUTS[i].x;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan2, INPUTS[i],
                                   LIBC_NAMESPACE::atan2f16(-x, y),
                                   TOLERANCE + 0.5);
    INPUTS[i].y = -INPUTS[i].y;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan2, INPUTS[i],
                                   LIBC_NAMESPACE::atan2f16(-x, -y),
                                   TOLERANCE + 0.5);
    INPUTS[i].x = -INPUTS[i].x;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan2, INPUTS[i],
                                   LIBC_NAMESPACE::atan2f16(x, -y),
                                   TOLERANCE + 0.5);
  }
}

TEST_F(LlvmLibcAtan2f16Test, InFloat16Range) {
  constexpr uint16_t X_START = FPBits(static_cast<float16>(0.25f)).uintval();
  constexpr uint16_t X_STOP = FPBits(static_cast<float16>(4.0f)).uintval();
  constexpr uint32_t X_COUNT = 1'23;
  constexpr uint32_t X_STEP = (X_STOP - X_START) / X_COUNT;

  constexpr uint16_t Y_START = FPBits(static_cast<float16>(0.25f)).uintval();
  constexpr uint16_t Y_STOP = FPBits(static_cast<float16>(4.0f)).uintval();
  constexpr uint32_t Y_COUNT = 1'37;
  constexpr uint32_t Y_STEP = (Y_STOP - Y_START) / Y_COUNT;

  auto test = [&](mpfr::RoundingMode rounding_mode) {
    mpfr::ForceRoundingMode __r(rounding_mode);
    if (!__r.success)
      return;

    uint64_t fails = 0;
    uint64_t finite_count = 0;
    uint64_t total_count = 0;
    float16 failed_x = 0.0f, failed_y = 0.0f, failed_r = 0.0f;
    double tol = 0.5;

    for (uint32_t i = 0; i <= X_COUNT; ++i) {
      uint16_t v = static_cast<uint16_t>(X_START + i * X_STEP);
      float16 x = FPBits(v).get_val();
      if (FPBits(v).is_nan() || FPBits(v).is_inf() || x < 0.0f)
        continue;

      for (uint32_t j = 0; j <= Y_COUNT; ++j) {
        uint16_t w = static_cast<uint16_t>(Y_START + j * Y_STEP);
        float16 y = FPBits(w).get_val();
        if (FPBits(w).is_nan() || FPBits(w).is_inf())
          continue;

        libc_errno = 0;
        float16 result = LIBC_NAMESPACE::atan2f16(x, y);
        ++total_count;
        if (FPBits(result).is_nan() || FPBits(result).is_inf())
          continue;

        ++finite_count;
        mpfr::BinaryInput<float16> input{x, y};

        if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Atan2, input,
                                               result, 0.5, rounding_mode)) {
          ++fails;
          while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(
              mpfr::Operation::Atan2, input, result, tol, rounding_mode)) {
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
      tlog << " Atan2f16 failed: " << fails << "/" << finite_count << "/"
           << total_count << " tests.\n"
           << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
    }
    if (fails) {
      mpfr::BinaryInput<float16> input{failed_x, failed_y};
      EXPECT_MPFR_MATCH(mpfr::Operation::Atan2, input, failed_r, 0.5,
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
