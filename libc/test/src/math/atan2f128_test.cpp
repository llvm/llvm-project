//===-- Unittests for atan2f128 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atan2f128.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcAtan2f128Test = LIBC_NAMESPACE::testing::FPTest<float128>;
using LIBC_NAMESPACE::testing::tlog;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcAtan2f128Test, InQuadRange) {
  constexpr StorageType X_COUNT = 123;
  constexpr StorageType X_START =
      FPBits(static_cast<float128>(0.25q)).uintval();
  constexpr StorageType X_STOP = FPBits(static_cast<float128>(4.0q)).uintval();
  constexpr StorageType X_STEP = (X_STOP - X_START) / X_COUNT;

  constexpr StorageType Y_COUNT = 137;
  constexpr StorageType Y_START =
      FPBits(static_cast<float128>(0.25q)).uintval();
  constexpr StorageType Y_STOP = FPBits(static_cast<float128>(4.0q)).uintval();
  constexpr StorageType Y_STEP = (Y_STOP - Y_START) / Y_COUNT;

  auto test = [&](mpfr::RoundingMode rounding_mode) {
    mpfr::ForceRoundingMode __r(rounding_mode);
    if (!__r.success)
      return;

    uint64_t fails = 0;
    uint64_t finite_count = 0;
    uint64_t total_count = 0;
    float128 failed_x = 0.0, failed_y = 0.0, failed_r = 0.0;
    double tol = 0.5;

    for (StorageType i = 0, v = X_START; i <= X_COUNT; ++i, v += X_STEP) {
      float128 x = FPBits(v).get_val();
      if (FPBits(x).is_inf_or_nan() || x < 0.0q)
        continue;

      for (StorageType j = 0, w = Y_START; j <= Y_COUNT; ++j, w += Y_STEP) {
        float128 y = FPBits(w).get_val();
        if (FPBits(y).is_inf_or_nan())
          continue;

        float128 result = LIBC_NAMESPACE::atan2f128(x, y);
        ++total_count;
        if (FPBits(result).is_inf_or_nan())
          continue;

        ++finite_count;
        mpfr::BinaryInput<float128> inputs{x, y};

        if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Atan2, inputs,
                                               result, 2.0, rounding_mode)) {
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
      tlog << " Atan2 failed: " << fails << "/" << finite_count << "/"
           << total_count << " tests.\n"
           << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
    }
    if (fails) {
      mpfr::BinaryInput<float128> inputs{failed_x, failed_y};
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
