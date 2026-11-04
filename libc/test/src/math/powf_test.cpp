//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for powf.
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/powf_double_eval.h"
#include "src/__support/math/powf_float_eval.h"
#include "src/math/powf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 1
#else
#define TOLERANCE 0
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

using LIBC_NAMESPACE::testing::tlog;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

class PowfTest : public LIBC_NAMESPACE::testing::FPTest<float> {
public:
  void test_tricky_inputs(float (*func)(float, float), double ulp_tolerance,
                          bool all_rounding) {
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
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Pow, INPUTS[i],
                                       func(INPUTS[i].x, INPUTS[i].y),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Pow, INPUTS[i],
                          func(INPUTS[i].x, INPUTS[i].y), ulp_tolerance);
      }
    }
  }

  void test_in_range(float (*func)(float, float), double ulp_tolerance,
                     bool all_rounding) {
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
      float mx = 0.0f, my = 0.0f, mr = 0.0f;
      double tol = ulp_tolerance;

      for (uint32_t i = 0, v = X_START; i <= X_COUNT; ++i, v += X_STEP) {
        float x = FPBits(v).get_val();
        if (FPBits(v).is_nan() || FPBits(v).is_inf() || x < 0.0f)
          continue;

        for (uint32_t j = 0, w = Y_START; j <= Y_COUNT; ++j, w += Y_STEP) {
          float y = FPBits(w).get_val();
          if (FPBits(w).is_nan() || FPBits(w).is_inf())
            continue;

          libc_errno = 0;
          float result = func(x, y);
          ++cc;
          if (FPBits(result).is_nan() || FPBits(result).is_inf())
            continue;

          ++count;
          mpfr::BinaryInput<float> inputs{x, y};

          if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Pow, inputs,
                                                 result, ulp_tolerance,
                                                 rounding_mode)) {
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
             << "   Max ULPs is at most: " << static_cast<uint64_t>(tol)
             << ".\n";
      }
      if (fails) {
        mpfr::BinaryInput<float> inputs{mx, my};
        EXPECT_MPFR_MATCH(mpfr::Operation::Pow, inputs, mr, ulp_tolerance,
                          rounding_mode);
      }
    };

    tlog << " Test Rounding To Nearest...\n";
    test(mpfr::RoundingMode::Nearest);

#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
    if (all_rounding) {
      tlog << " Test Rounding Downward...\n";
      test(mpfr::RoundingMode::Downward);

      tlog << " Test Rounding Upward...\n";
      test(mpfr::RoundingMode::Upward);

      tlog << " Test Rounding Toward Zero...\n";
      test(mpfr::RoundingMode::TowardZero);
    }
#endif // LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  }
};

#define LIST_POWF_TESTS(suffix, func, tricky_ulp, range_ulp, all_rounding)     \
  using LlvmLibcPowfTest##suffix = PowfTest;                                   \
  TEST_F(LlvmLibcPowfTest##suffix, TrickyInputs) {                             \
    test_tricky_inputs(&func, tricky_ulp, all_rounding);                       \
  }                                                                            \
  TEST_F(LlvmLibcPowfTest##suffix, InFloatRange) {                             \
    test_in_range(&func, range_ulp, all_rounding);                             \
  }                                                                            \
  static_assert(true, "Require semicolon.")

LIST_POWF_TESTS(Default, LIBC_NAMESPACE::powf,
                /*tricky_ulp=*/TOLERANCE + 0.5, /*range_ulp=*/TOLERANCE + 0.5,
                /*all_rounding=*/true);
LIST_POWF_TESTS(DoubleEval, LIBC_NAMESPACE::math::double_eval::powf,
                /*tricky_ulp=*/TOLERANCE + 0.5, /*range_ulp=*/TOLERANCE + 0.5,
                /*all_rounding=*/true);
LIST_POWF_TESTS(FloatEval, LIBC_NAMESPACE::math::float_eval::powf,
                /*tricky_ulp=*/1.0, /*range_ulp=*/1.5,
                /*all_rounding=*/false);
