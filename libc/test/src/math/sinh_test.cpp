//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for sinh.
///
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/sinh_double_eval.h"
#include "src/math/sinh.h"
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

LIBC_NOINLINE double sinh_double_eval(double x) {
  return LIBC_NAMESPACE::math::double_eval::sinh(x);
}

class SinhTest : public LIBC_NAMESPACE::testing::FPTest<double> {
public:
  void test_special_numbers(double (*func)(double)) {
    EXPECT_FP_EQ(aNaN, func(aNaN));
    EXPECT_FP_EQ(inf, func(inf));
    EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, func(neg_inf));
    EXPECT_FP_EQ_ALL_ROUNDING(0.0, func(0.0));
    EXPECT_FP_EQ_ALL_ROUNDING(-0.0, func(-0.0));
  }

  void test_overflow(double (*func)(double)) {
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(0x1.0p20), FE_OVERFLOW);
    EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(-0x1.0p20), FE_OVERFLOW);
  }

  void test_tricky_inputs(double (*func)(double), double ulp_tolerance = 0.5,
                          bool all_rounding = true) {
    constexpr int N = 16;
    constexpr uint64_t INPUTS[N] = {
        0x3e50'0000'0000'0000ULL, // x = 0x1.0p-26
        0x3e50'0000'0000'0001ULL, // x = 0x1.0000000000001p-26
        0x3e6b'cee7'0ebe'7ec9ULL, // x = 0x1.bcee70ebe7ec9p-25
        0x3f18'a62b'e5b5'f93cULL, // x = 0x1.8a62be5b5f93cp-14
        0x3f8f'ffff'ffff'ffffULL, // x = 0x1.fffffffffffffp-7
        0x3fb3'30b5'a711'0cf0ULL, // x = 0x1.330b5a7110cfp-4
        0x3fcf'ffff'ffff'ffffULL, // x = 0x1.fffffffffffffp-3
        0x3fd0'0000'0000'0000ULL, // x = 0x1.0p-2
        0x3fd3'b6db'f12c'e919ULL, // x = 0x1.3b6dbf12ce919p-2
        0x3fe6'2e42'fefa'39efULL, // x = 0x1.62e42fefa39efp-1
        0x3ff7'4385'446d'71c3ULL, // x = 0x1.74385446d71c3p+0
        0x401c'ac90'6a4f'1a46ULL, // x = 0x1.cac906a4f1a46p+2
        0x4032'57b9'12e2'b431ULL, // x = 0x1.257b912e2b431p+4
        0x4086'2e42'fefa'39efULL, // x = 0x1.62e42fefa39efp+9
        0x4086'33ce'8fb9'f87dULL, // x = 0x1.633ce8fb9f87dp+9
        0x4086'33ce'8fb9'f87eULL, // x = 0x1.633ce8fb9f87ep+9
    };
    for (int i = 0; i < N; ++i) {
      double x = FPBits(INPUTS[i]).get_val();
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinh, x, func(x),
                                       ulp_tolerance);
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinh, -x, func(-x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Sinh, x, func(x), ulp_tolerance);
        EXPECT_MPFR_MATCH(mpfr::Operation::Sinh, -x, func(-x), ulp_tolerance);
      }
    }
  }

  void test_in_range(double (*func)(double), double ulp_tolerance = 0.5,
                     bool all_rounding = true) {
    constexpr uint64_t COUNT = 1'231;
    constexpr uint64_t START = FPBits(0x1.0p-30).uintval();
    constexpr uint64_t STOP = FPBits(710.0).uintval();
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
        if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Sinh, x, result,
                                               ulp_tolerance, rounding_mode)) {
          ++fails;
          while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(
              mpfr::Operation::Sinh, x, result, tol, rounding_mode)) {
            mx = x;
            mr = result;

            if (tol > 1000.0)
              break;

            tol *= 2.0;
          }
        }
      }
      if (fails) {
        tlog << " Sinh failed: " << fails << "/" << count << "/" << cc
             << " tests.\n";
        tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol)
             << ".\n";
        EXPECT_MPFR_MATCH(mpfr::Operation::Sinh, mx, mr, ulp_tolerance,
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

#define LIST_SINH_TESTS(suffix, func, ulp_tolerance, all_rounding)             \
  using LlvmLibcSinhTest##suffix = SinhTest;                                   \
  TEST_F(LlvmLibcSinhTest##suffix, SpecialNumbers) {                           \
    test_special_numbers(&func);                                               \
  }                                                                            \
  TEST_F(LlvmLibcSinhTest##suffix, Overflow) { test_overflow(&func); }         \
  TEST_F(LlvmLibcSinhTest##suffix, TrickyInputs) {                             \
    test_tricky_inputs(&func, ulp_tolerance, all_rounding);                    \
  }                                                                            \
  TEST_F(LlvmLibcSinhTest##suffix, InDoubleRange) {                            \
    test_in_range(&func, ulp_tolerance, all_rounding);                         \
  }                                                                            \
  static_assert(true, "Require semicolon.")

LIST_SINH_TESTS(Default, LIBC_NAMESPACE::sinh,
                /*ulp_tolerance=*/TOLERANCE + 0.5,
                /*all_rounding=*/ALL_ROUNDING);
LIST_SINH_TESTS(DoubleEval, sinh_double_eval,
                /*ulp_tolerance=*/TOLERANCE + 0.5,
                /*all_rounding=*/ALL_ROUNDING);
