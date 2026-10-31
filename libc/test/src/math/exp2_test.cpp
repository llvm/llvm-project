//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for exp2.
///
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/exp2_double_eval.h"
#include "src/math/exp2.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#if defined(LIBC_MATH_HAS_SMALL_TABLES) &&                                     \
    !defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY)
#define TOLERANCE 2
#else // LIBC_MATH_HAS_SMALL_TABLE && LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
#define TOLERANCE 1
#endif // LIBC_MATH_HAS_SMALL_TABLES && !LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
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

LIBC_NOINLINE double exp2_double_eval(double x) {
  return LIBC_NAMESPACE::math::double_eval::exp2(x);
}

class Exp2Test : public LIBC_NAMESPACE::testing::FPTest<double> {
public:
  void test_special_numbers(double (*func)(double)) {
    EXPECT_FP_EQ(aNaN, func(aNaN));
    EXPECT_FP_EQ(inf, func(inf));
    EXPECT_FP_EQ_ALL_ROUNDING(zero, func(neg_inf));
    EXPECT_FP_EQ_ALL_ROUNDING(1.0, func(0.0));
    EXPECT_FP_EQ_ALL_ROUNDING(1.0, func(-0.0));
  }

  void test_overflow(double (*func)(double)) {
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(0x1.0p20), FE_OVERFLOW);
  }

  void test_underflow(double (*func)(double)) {
    EXPECT_FP_EQ_WITH_EXCEPTION(zero, func(-0x1.0p20), FE_UNDERFLOW);
  }

  void test_tricky_inputs(double (*func)(double), double ulp_tolerance = 0.5,
                          bool all_rounding = true) {
    constexpr int N = 15;
    constexpr uint64_t INPUTS[N] = {
        0x3fd79289c6e6a5c0, // x = 0x1.79289c6e6a5cp-2
        0x3fd05de80a173ea0, // x = 0x1.05de80a173eap-2
        0x3fda7b764e2cf47a, // x = 0x1.a7b764e2cf47ap-2
        0x4044c19e5712e377, // x = 0x1.4c19e5712e377p+5
        0xc039a74cdab36c28, // x = -0x1.9a74cdab36c28p+4
        0xc04757852a4b93aa, // x = -0x1.757852a4b93aap+5
        0xc085b3e4e2e3bba9, // x = -0x1.5b3e4e2e3bba9p+9
        0xc086960d591aec34, // x = -0x1.6960d591aec34p+9
        0xc086232c09d58d91, // x = -0x1.6232c09d58d91p+9
        0xc0874910d52d3051, // x = -0x1.74910d52d3051p+9
        0xc0867a172ceb0990, // x = -0x1.67a172ceb099p+9
        0xc08ff80000000000, // x = -0x1.ff8p+9
        0xbc971547652b82fe, // x = -0x1.71547652b82fep-54
        0xbf1eb7a4cb841fcc, // x = -0x1.eb7a4cb841fccp-14
        0xbf19a61fb925970d, // x = -0x1.9a61fb925970dp-14
    };
    for (int i = 0; i < N; ++i) {
      double x = FPBits(INPUTS[i]).get_val();
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp2, x, func(x), ulp_tolerance);
      }
    }
  }

  void test_in_range(double (*func)(double), double ulp_tolerance = 0.5,
                     bool all_rounding = true) {
    constexpr uint64_t COUNT = 1'231;
    constexpr uint64_t START = FPBits(0.25).uintval();
    constexpr uint64_t STOP = FPBits(4.0).uintval();
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
        if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Exp2, x, result,
                                               ulp_tolerance, rounding_mode)) {
          ++fails;
          while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(
              mpfr::Operation::Exp2, x, result, tol, rounding_mode)) {
            mx = x;
            mr = result;

            if (tol > 1000.0)
              break;

            tol *= 2.0;
          }
        }
      }
      if (fails) {
        tlog << " Exp2 failed: " << fails << "/" << count << "/" << cc
             << " tests.\n";
        tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol)
             << ".\n";
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp2, mx, mr, ulp_tolerance,
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

#define LIST_EXP2_TESTS(suffix, func, ulp_tolerance, all_rounding)             \
  using LlvmLibcExp2Test##suffix = Exp2Test;                                   \
  TEST_F(LlvmLibcExp2Test##suffix, SpecialNumbers) {                           \
    test_special_numbers(&func);                                               \
  }                                                                            \
  TEST_F(LlvmLibcExp2Test##suffix, Overflow) { test_overflow(&func); }         \
  TEST_F(LlvmLibcExp2Test##suffix, Underflow) { test_underflow(&func); }       \
  TEST_F(LlvmLibcExp2Test##suffix, TrickyInputs) {                             \
    test_tricky_inputs(&func, ulp_tolerance, all_rounding);                    \
  }                                                                            \
  TEST_F(LlvmLibcExp2Test##suffix, InDoubleRange) {                            \
    test_in_range(&func, ulp_tolerance, all_rounding);                         \
  }                                                                            \
  static_assert(true, "Require semicolon.")

LIST_EXP2_TESTS(Default, LIBC_NAMESPACE::exp2,
                /*ulp_tolerance=*/TOLERANCE + 0.5,
                /*all_rounding=*/ALL_ROUNDING);
LIST_EXP2_TESTS(DoubleEval, exp2_double_eval,
                /*ulp_tolerance=*/TOLERANCE + 0.5,
                /*all_rounding=*/ALL_ROUNDING);
