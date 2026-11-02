//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for tanh.
///
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/tanh_double_eval.h"
#include "src/math/tanh.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
#define TOLERANCE 2
#else // LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
#define TOLERANCE 1
#endif // !LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
#else  // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 0
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
#define ALL_ROUNDING false
#else
#define ALL_ROUNDING true
#endif // LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
using LIBC_NAMESPACE::testing::tlog;

LIBC_NOINLINE double tanh_double_eval(double x) {
  return LIBC_NAMESPACE::math::double_eval::tanh(x);
}

class TanhTest : public LIBC_NAMESPACE::testing::FPTest<double> {
public:
  void test_special_numbers(double (*func)(double)) {
    EXPECT_FP_EQ(aNaN, func(aNaN));
    EXPECT_FP_EQ_ALL_ROUNDING(1.0, func(inf));
    EXPECT_FP_EQ_ALL_ROUNDING(-1.0, func(neg_inf));
    EXPECT_FP_EQ_ALL_ROUNDING(0.0, func(0.0));
    EXPECT_FP_EQ_ALL_ROUNDING(-0.0, func(-0.0));
  }

  void test_tricky_inputs(double (*func)(double), double ulp_tolerance = 0.5,
                          bool all_rounding = true) {
    constexpr int N = 22;
    constexpr uint64_t INPUTS[N] = {
        0x3e40'0000'0000'0000ULL, // x = 0x1.0p-27
        0x3e40'0000'0000'0001ULL, // x = 0x1.0000000000001p-27
        0x3e4d'12ed'0af1'a27fULL, // x = 0x1.d12ed0af1a27fp-27
        0x3fbf'faa0'8fca'd74eULL, // x = 0x1.ffaa08fcad74ep-5
        0x3fbf'ffff'ffff'ffffULL, // x = 0x1.fffffffffffffp-4
        0x3fc0'0000'0000'0000ULL, // x = 0x1.0p-3
        0x3fc0'0452'4207'fad9ULL, // x = 0x1.004524207fad9p-3
        0x3fca'c343'b179'fec4ULL, // x = 0x1.ac343b179fec4p-3
        0x3fd0'0764'a988'bf73ULL, // x = 0x1.00764a988bf73p-2
        0x3fd1'7d1e'8a63'711fULL, // x = 0x1.17d1e8a63711fp-2
        0x3fd2'91c6'01a0'5276ULL, // x = 0x1.291c601a05276p-2
        0x3fd3'6f33'd51c'264dULL, // x = 0x1.36f33d51c264dp-2
        0x3fd4'3eae'a236'49c3ULL, // x = 0x1.43eaea23649c3p-2
        0x3fdd'88d7'550b'2826ULL, // x = 0x1.d88d7550b2826p-2
        0x3fde'611a'a58a'b608ULL, // x = 0x1.e611aa58ab608p-2
        0x3fe0'1efe'7ac8'c15dULL, // x = 0x1.01efe7ac8c15dp-1
        0x3fe1'005e'c0bc'cabbULL, // x = 0x1.1005ec0bccabbp-1
        0x3fe3'3dfe'b0fa'4bfeULL, // x = 0x1.33dfeb0fa4bfep-1
        0x3fe4'9f24'ac5c'ac35ULL, // x = 0x1.49f24ac5cac35p-1
        0x4010'9cc2'de69'e78cULL, // x = 0x1.09cc2de69e78cp+2
        0x4033'0fc1'931f'09c9ULL, // x = 0x1.30fc1931f09c9p+4
        0x4033'0fc1'931f'09caULL, // x = 0x1.30fc1931f09cap+4
    };
    for (int i = 0; i < N; ++i) {
      double x = FPBits(INPUTS[i]).get_val();
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Tanh, x, func(x),
                                       ulp_tolerance);
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Tanh, -x, func(-x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Tanh, x, func(x), ulp_tolerance);
        EXPECT_MPFR_MATCH(mpfr::Operation::Tanh, -x, func(-x), ulp_tolerance);
      }
    }
  }

  void test_in_range(double (*func)(double), double ulp_tolerance = 0.5,
                     bool all_rounding = true) {
    constexpr uint64_t COUNT = 1'231;
    constexpr uint64_t START = FPBits(0x1.0p-30).uintval();
    constexpr uint64_t STOP = FPBits(20.0).uintval();
    constexpr uint64_t STEP = (STOP - START) / COUNT;

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
        if (FPBits(v).is_nan() || FPBits(v).is_inf() || x < 0.0)
          continue;
        double result = func(x);
        ++cc;
        if (FPBits(result).is_nan() || FPBits(result).is_inf())
          continue;

        ++count;
        if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Tanh, x, result,
                                               ulp_tolerance, rounding_mode)) {
          ++fails;
          while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(
              mpfr::Operation::Tanh, x, result, tol, rounding_mode)) {
            mx = x;
            mr = result;

            if (tol > 1000.0)
              break;

            tol *= 2.0;
          }
        }
      }
      if (fails) {
        tlog << " Tanh failed: " << fails << "/" << count << "/" << cc
             << " tests.\n";
        tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol)
             << ".\n";
        EXPECT_MPFR_MATCH(mpfr::Operation::Tanh, mx, mr, ulp_tolerance,
                          rounding_mode);
      }
    };

    tlog << " Test Rounding To Nearest...\n";
    test(mpfr::RoundingMode::Nearest);

    if (all_rounding) {
      tlog << " Test Rounding Downward...\n";
      test(mpfr::RoundingMode::Downward);

      tlog << " Test Rounding Upward...\n";
      test(mpfr::RoundingMode::Upward);

      tlog << " Test Rounding Toward Zero...\n";
      test(mpfr::RoundingMode::TowardZero);
    }
  }
};

#define LIST_TANH_TESTS(suffix, func, ulp_tolerance, all_rounding)             \
  using LlvmLibcTanhTest##suffix = TanhTest;                                   \
  TEST_F(LlvmLibcTanhTest##suffix, SpecialNumbers) {                           \
    test_special_numbers(&func);                                               \
  }                                                                            \
  TEST_F(LlvmLibcTanhTest##suffix, TrickyInputs) {                             \
    test_tricky_inputs(&func, ulp_tolerance, all_rounding);                    \
  }                                                                            \
  TEST_F(LlvmLibcTanhTest##suffix, InDoubleRange) {                            \
    test_in_range(&func, ulp_tolerance, all_rounding);                         \
  }                                                                            \
  static_assert(true, "Require semicolon.")

LIST_TANH_TESTS(Default, LIBC_NAMESPACE::tanh,
                /*ulp_tolerance=*/TOLERANCE + 0.5,
                /*all_rounding=*/ALL_ROUNDING);
LIST_TANH_TESTS(DoubleEval, tanh_double_eval,
                /*ulp_tolerance=*/TOLERANCE + 0.5,
                /*all_rounding=*/ALL_ROUNDING);
