//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for cosh.
///
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/cosh_double_eval.h"
#include "src/math/cosh.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 1
#else // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 0
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
#define ALL_ROUNDING false
#else
#define ALL_ROUNDING true
#endif // LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
using LIBC_NAMESPACE::testing::tlog;

LIBC_NOINLINE double cosh_double_eval(double x) {
  return LIBC_NAMESPACE::math::double_eval::cosh(x);
}

class CoshTest : public LIBC_NAMESPACE::testing::FPTest<double> {
public:
  void test_special_numbers(double (*func)(double)) {
    EXPECT_FP_EQ(aNaN, func(aNaN));
    EXPECT_FP_EQ(inf, func(inf));
    EXPECT_FP_EQ_ALL_ROUNDING(inf, func(neg_inf));
    EXPECT_FP_EQ_ALL_ROUNDING(1.0, func(0.0));
    EXPECT_FP_EQ_ALL_ROUNDING(1.0, func(-0.0));
  }

  void test_overflow(double (*func)(double)) {
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(0x1.0p20), FE_OVERFLOW);
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(-0x1.0p20), FE_OVERFLOW);
  }

  void test_tricky_inputs(double (*func)(double), double ulp_tolerance = 0.5,
                          bool all_rounding = true) {
    constexpr int N = 20;
    constexpr uint64_t INPUTS[N] = {
        0x3e40'0000'0000'0000ULL, // x = 0x1.0p-27
        0x3e40'0000'0000'0001ULL, // x = 0x1.0000000000001p-27
        0x3eb0'1fe0'3f61'baceULL, // x = 0x1.01fe03f61bacep-20
        0x3f10'1218'4c8c'9313ULL, // x = 0x1.012184c8c9313p-14
        0x3f80'1813'9e73'ee47ULL, // x = 0x1.018139e73ee47p-7
        0x3fa0'10b2'a872'e30bULL, // x = 0x1.010b2a872e30bp-5
        0x3fb9'a5e3'cbe1'985eULL, // x = 0x1.9a5e3cbe1985ep-4
        0x3fc5'2a11'832e'847dULL, // x = 0x1.52a11832e847dp-3
        0x3fcb'f030'5e2c'6c37ULL, // x = 0x1.bf0305e2c6c37p-3
        0x3fd1'7326'ffc0'9f68ULL, // x = 0x1.17326ffc09f68p-2
        0x3fd3'd27b'f16d'8bdbULL, // x = 0x1.3d27bf16d8bdbp-2
        0x3fe0'3923'f2b4'7c07ULL, // x = 0x1.03923f2b47c07p-1
        0x3fea'6031'cd5f'93baULL, // x = 0x1.a6031cd5f93bap-1
        0x3ff1'04b6'48f1'13a1ULL, // x = 0x1.104b648f113a1p+0
        0x3ff1'5857'20f3'5cd9ULL, // x = 0x1.1585720f35cd9p+0
        0x3ffe'9cc7'ed2e'1a7eULL, // x = 0x1.e9cc7ed2e1a7ep+0
        0x4004'3180'ea85'4696ULL, // x = 0x1.43180ea854696p+1
        0x4017'2581'1dcf'6782ULL, // x = 0x1.725811dcf6782p+2
        0x4086'33ce'8fb9'f87dULL, // x = 0x1.633ce8fb9f87dp+9
        0x4086'33ce'8fb9'f87eULL, // x = 0x1.633ce8fb9f87ep+9
    };
    for (int i = 0; i < N; ++i) {
      double x = FPBits(INPUTS[i]).get_val();
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cosh, x, func(x),
                                       ulp_tolerance);
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cosh, -x, func(-x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Cosh, x, func(x), ulp_tolerance);
        EXPECT_MPFR_MATCH(mpfr::Operation::Cosh, -x, func(-x), ulp_tolerance);
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
        if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Cosh, x, result,
                                               ulp_tolerance, rounding_mode)) {
          ++fails;
          while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(
              mpfr::Operation::Cosh, x, result, tol, rounding_mode)) {
            mx = x;
            mr = result;

            if (tol > 1000.0)
              break;

            tol *= 2.0;
          }
        }
      }
      if (fails) {
        tlog << " Cosh failed: " << fails << "/" << count << "/" << cc
             << " tests.\n";
        tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol)
             << ".\n";
        EXPECT_MPFR_MATCH(mpfr::Operation::Cosh, mx, mr, ulp_tolerance,
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

#define LIST_COSH_TESTS(suffix, func, ulp_tolerance, all_rounding)             \
  using LlvmLibcCoshTest##suffix = CoshTest;                                   \
  TEST_F(LlvmLibcCoshTest##suffix, SpecialNumbers) {                           \
    test_special_numbers(&func);                                               \
  }                                                                            \
  TEST_F(LlvmLibcCoshTest##suffix, Overflow) { test_overflow(&func); }         \
  TEST_F(LlvmLibcCoshTest##suffix, TrickyInputs) {                             \
    test_tricky_inputs(&func, ulp_tolerance, all_rounding);                    \
  }                                                                            \
  TEST_F(LlvmLibcCoshTest##suffix, InDoubleRange) {                            \
    test_in_range(&func, ulp_tolerance, all_rounding);                         \
  }                                                                            \
  static_assert(true, "Require semicolon.")

LIST_COSH_TESTS(Default, LIBC_NAMESPACE::cosh,
                /*ulp_tolerance=*/TOLERANCE + 0.5,
                /*all_rounding=*/ALL_ROUNDING);
LIST_COSH_TESTS(DoubleEval, cosh_double_eval,
                /*ulp_tolerance=*/TOLERANCE + 0.5,
                /*all_rounding=*/ALL_ROUNDING);
