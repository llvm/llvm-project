//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for exp10.
///
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/exp10_double_eval.h"
#include "src/math/exp10.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#if defined(LIBC_MATH_HAS_SMALL_TABLES) &&                                     \
    !defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY)
#define TOLERANCE 2
#else
#define TOLERANCE 1
#endif // LIBC_MATH_HAS_SMALL_TABLES && !LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
#else
#define TOLERANCE 0
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
#define ALL_ROUNDING false
#else
#define ALL_ROUNDING true
#endif // LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
using LIBC_NAMESPACE::testing::tlog;

LIBC_NOINLINE double exp10_double_eval(double x) {
  return LIBC_NAMESPACE::math::double_eval::exp10(x);
}

class Exp10Test : public LIBC_NAMESPACE::testing::FPTest<double> {
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
    constexpr int N = 41;
    constexpr uint64_t INPUTS[N] = {
        0x40033093317082F8, 0x3FD79289C6E6A5C0,
        0x3FD05DE80A173EA0, // 0x1.05de80a173eap-2
        0xbf1eb7a4cb841fcc, // -0x1.eb7a4cb841fccp-14
        0xbf19a61fb925970d,
        0x3fda7b764e2cf47a, // 0x1.a7b764e2cf47ap-2
        0xc04757852a4b93aa, // -0x1.757852a4b93aap+5
        0x4044c19e5712e377, // x=0x1.4c19e5712e377p+5
        0xbf19a61fb925970d, // x=-0x1.9a61fb925970dp-14
        0xc039a74cdab36c28, // x=-0x1.9a74cdab36c28p+4
        0xc085b3e4e2e3bba9, // x=-0x1.5b3e4e2e3bba9p+9
        0xc086960d591aec34, // x=-0x1.6960d591aec34p+9
        0xc086232c09d58d91, // x=-0x1.6232c09d58d91p+9
        0xc0874910d52d3051, // x=-0x1.74910d52d3051p9
        0xc0867a172ceb0990, // x=-0x1.67a172ceb099p+9
        0xc08ff80000000000, // x=-0x1.ff8p+9
        0xbc971547652b82fe, // x=-0x1.71547652b82fep-54
        0x0000000000000000, // x = 0
        0x3ff0000000000000, // x = 1
        0x4000000000000000, // x = 2
        0x4008000000000000, // x = 3
        0x4010000000000000, // x = 4
        0x4014000000000000, // x = 5
        0x4018000000000000, // x = 6
        0x401c000000000000, // x = 7
        0x4020000000000000, // x = 8
        0x4022000000000000, // x = 9
        0x4024000000000000, // x = 10
        0x4026000000000000, // x = 11
        0x4028000000000000, // x = 12
        0x402a000000000000, // x = 13
        0x402c000000000000, // x = 14
        0x402e000000000000, // x = 15
        0x4030000000000000, // x = 16
        0x4031000000000000, // x = 17
        0x4032000000000000, // x = 18
        0x4033000000000000, // x = 19
        0x4034000000000000, // x = 20
        0x4035000000000000, // x = 21
        0x4036000000000000, // x = 22
        0x4037000000000000, // x = 23
    };
    for (int i = 0; i < N; ++i) {
      double x = FPBits(INPUTS[i]).get_val();
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp10, x, func(x), ulp_tolerance);
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
        if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Exp10, x,
                                               result, ulp_tolerance,
                                               rounding_mode)) {
          ++fails;
          while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(
              mpfr::Operation::Exp10, x, result, tol, rounding_mode)) {
            mx = x;
            mr = result;

            if (tol > 1000.0)
              break;

            tol *= 2.0;
          }
        }
      }
      if (fails) {
        tlog << " Exp10 failed: " << fails << "/" << count << "/" << cc
             << " tests.\n";
        tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol)
             << ".\n";
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp10, mx, mr, ulp_tolerance,
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

#define LIST_EXP10_TESTS(suffix, func, ulp_tolerance, all_rounding)            \
  using LlvmLibcExp10Test##suffix = Exp10Test;                                 \
  TEST_F(LlvmLibcExp10Test##suffix, SpecialNumbers) {                          \
    test_special_numbers(&func);                                               \
  }                                                                            \
  TEST_F(LlvmLibcExp10Test##suffix, Overflow) { test_overflow(&func); }        \
  TEST_F(LlvmLibcExp10Test##suffix, Underflow) { test_underflow(&func); }      \
  TEST_F(LlvmLibcExp10Test##suffix, TrickyInputs) {                            \
    test_tricky_inputs(&func, ulp_tolerance, all_rounding);                    \
  }                                                                            \
  TEST_F(LlvmLibcExp10Test##suffix, InDoubleRange) {                           \
    test_in_range(&func, ulp_tolerance, all_rounding);                         \
  }                                                                            \
  static_assert(true, "Require semicolon.")

LIST_EXP10_TESTS(Default, LIBC_NAMESPACE::exp10,
                 /*ulp_tolerance=*/TOLERANCE + 0.5,
                 /*all_rounding=*/ALL_ROUNDING);
LIST_EXP10_TESTS(DoubleEval, exp10_double_eval,
                 /*ulp_tolerance=*/TOLERANCE + 0.5,
                 /*all_rounding=*/ALL_ROUNDING);
