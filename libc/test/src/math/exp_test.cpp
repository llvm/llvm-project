//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for exp.
///
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/exp_double_eval.h"
#include "src/__support/math/exp_integer_eval.h"
#include "src/math/exp.h"
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

LIBC_NOINLINE double exp_double_eval(double x) {
  return LIBC_NAMESPACE::math::double_eval::exp(x);
}

LIBC_NOINLINE double exp_integer_eval(double x) {
  return LIBC_NAMESPACE::math::integer_eval::exp(x);
}

class ExpTest : public LIBC_NAMESPACE::testing::FPTest<double> {
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
    constexpr int N = 18;
    constexpr uint64_t INPUTS[N] = {
        0x3ca0000000000000, // x = 0x1.0p-53
        0xbca0000000000000, // x = -0x1.0p-53
        0xbcc0000000000001, // x = -0x1.0000000000001p-51
        0xbd26000000000079, // x = -0x1.6000000000079p-45
        0x3fd79289c6e6a5c0, // x = 0x1.79289c6e6a5cp-2
        0x3fd05de80a173ea0, // x = 0x1.05de80a173eap-2
        0x3fda7b764e2cf47a, // x = 0x1.a7b764e2cf47ap-2
        0x4044c19e5712e377, // x = 0x1.4c19e5712e377p+5
        0x40862e42fefa39ef, // x = 0x1.62e42fefa39efp+9
        0xc039a74cdab36c28, // x = -0x1.9a74cdab36c28p+4
        0xc04757852a4b93aa, // x = -0x1.757852a4b93aap+5
        0xc085b3e4e2e3bba9, // x = -0x1.5b3e4e2e3bba9p+9
        0xc086960d591aec34, // x = -0x1.6960d591aec34p+9
        0xc086232c09d58d91, // x = -0x1.6232c09d58d91p+9
        0xc0874910d52d3051, // x = -0x1.74910d52d3051p+9
        0xc0867a172ceb0990, // x = -0x1.67a172ceb099p+9
        0xbf1eb7a4cb841fcc, // x = -0x1.eb7a4cb841fccp-14
        0xbf19a61fb925970d, // x = -0x1.9a61fb925970dp-14
    };
    for (int i = 0; i < N; ++i) {
      double x = FPBits(INPUTS[i]).get_val();
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp, x, func(x), ulp_tolerance);
      }
    }
  }

  // Probe adjacent representable inputs on both sides of transition points.
  void test_transition_neighbors(double (*func)(double), double ulp_tolerance,
                                 bool all_rounding) {
    constexpr uint64_t CENTERS[] = {
        0x3ca0000000000000ULL, // +2^-53 tiny boundary
        0xbca0000000000000ULL, // -2^-53 tiny boundary
        0x4085e00000000000ULL, // +700 integer-kernel boundary
        0xc085e00000000000ULL, // -700 integer-kernel boundary
        0x40862e42fefa39f0ULL, // overflow boundary
        0xc0874910d52d3052ULL, // underflow-to-zero boundary
        0xc086232bdd7abcd2ULL, // log(minimum normal), rounded to binary64
        0xc0874385446d71c3ULL, // log(minimum subnormal), rounded to binary64
    };
    for (uint64_t center : CENTERS) {
      for (uint64_t offset = 0; offset < 5; ++offset) {
        double x = FPBits(center - 2 + offset).get_val();
        if (all_rounding) {
          EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, func(x),
                                         ulp_tolerance);
        } else {
          EXPECT_MPFR_MATCH(mpfr::Operation::Exp, x, func(x), ulp_tolerance);
        }
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
        if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Exp, x, result,
                                               ulp_tolerance, rounding_mode)) {
          ++fails;
          while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(
              mpfr::Operation::Exp, x, result, tol, rounding_mode)) {
            mx = x;
            mr = result;

            if (tol > 1000.0)
              break;

            tol *= 2.0;
          }
        }
      }
      if (fails) {
        tlog << " Exp failed: " << fails << "/" << count << "/" << cc
             << " tests.\n";
        tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol)
             << ".\n";
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp, mx, mr, ulp_tolerance,
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

#define LIST_EXP_TESTS(suffix, func, ulp_tolerance, all_rounding)              \
  using LlvmLibcExpTest##suffix = ExpTest;                                     \
  TEST_F(LlvmLibcExpTest##suffix, SpecialNumbers) {                            \
    test_special_numbers(&func);                                               \
  }                                                                            \
  TEST_F(LlvmLibcExpTest##suffix, Overflow) { test_overflow(&func); }          \
  TEST_F(LlvmLibcExpTest##suffix, Underflow) { test_underflow(&func); }        \
  TEST_F(LlvmLibcExpTest##suffix, TrickyInputs) {                              \
    test_tricky_inputs(&func, ulp_tolerance, all_rounding);                    \
  }                                                                            \
  TEST_F(LlvmLibcExpTest##suffix, InDoubleRange) {                             \
    test_in_range(&func, ulp_tolerance, all_rounding);                         \
  }                                                                            \
  TEST_F(LlvmLibcExpTest##suffix, TransitionNeighbors) {                       \
    test_transition_neighbors(&func, ulp_tolerance, all_rounding);             \
  }                                                                            \
  static_assert(true, "Require semicolon.")

LIST_EXP_TESTS(Default, LIBC_NAMESPACE::exp,
               /*ulp_tolerance=*/TOLERANCE + 0.5,
               /*all_rounding=*/ALL_ROUNDING);
LIST_EXP_TESTS(DoubleEval, exp_double_eval,
               /*ulp_tolerance=*/TOLERANCE + 0.5,
               /*all_rounding=*/ALL_ROUNDING);

LIST_EXP_TESTS(IntegerEval, exp_integer_eval,
               /*ulp_tolerance=*/TOLERANCE + 0.5,
               /*all_rounding=*/ALL_ROUNDING);

TEST_F(LlvmLibcExpTestIntegerEval, InexactZeroAndNaN) {
  LIBC_NAMESPACE::libc_errno = 123;
  EXPECT_FP_EQ_WITH_EXCEPTION(1.0, exp_integer_eval(0.0), 0);
  EXPECT_FP_EQ_WITH_EXCEPTION(1.0, exp_integer_eval(-0.0), 0);
  EXPECT_FP_EQ_WITH_EXCEPTION(FPBits(uint64_t(0x4005bf0a8b145769ULL)).get_val(),
                              exp_integer_eval(1.0), FE_INEXACT);
  EXPECT_EQ(int(LIBC_NAMESPACE::libc_errno), 123);
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, exp_integer_eval(inf), 0);
  EXPECT_FP_EQ_WITH_EXCEPTION(0.0, exp_integer_eval(neg_inf), 0);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(exp_integer_eval(aNaN), 0);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(exp_integer_eval(sNaN), FE_INVALID);
}

TEST_F(LlvmLibcExpTestIntegerEval, KernelAcceptanceAndFallback) {
  uint64_t output = 0;
  EXPECT_TRUE(LIBC_NAMESPACE::math::exp_integer_internal::exp64_candidate(
      FPBits(1.0).uintval(), 0, &output));
  EXPECT_EQ(output, uint64_t(0x4005bf0a8b145769ULL));
  EXPECT_FALSE(LIBC_NAMESPACE::math::exp_integer_internal::exp64_candidate(
      FPBits(0.0).uintval(), 0, &output));
  // Near this rounding midpoint, the Q64 enclosure cannot determine the result.
  // The public implementation must resolve it through the existing fallback.
  EXPECT_FALSE(LIBC_NAMESPACE::math::exp_integer_internal::exp64_candidate(
      FPBits(0x1.0p-53).uintval(), 0, &output));
  if (ALL_ROUNDING) {
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, 0x1.0p-53,
                                   exp_integer_eval(0x1.0p-53),
                                   TOLERANCE + 0.5);
  } else {
    EXPECT_MPFR_MATCH(mpfr::Operation::Exp, 0x1.0p-53,
                      exp_integer_eval(0x1.0p-53), TOLERANCE + 0.5);
  }
}

TEST_F(LlvmLibcExpTestIntegerEval, KernelRefusalPreservesOutput) {
  namespace internal = LIBC_NAMESPACE::math::exp_integer_internal;
  constexpr uint64_t SENTINEL = UINT64_MAX;
  const uint64_t inputs[] = {
      0,
      0x8000000000000000ULL,
      1,
      0x8000000000000001ULL,
      0x4085e00000000000ULL,
      0xc085e00000000000ULL,
      0x7ff0000000000000ULL,
      0xfff0000000000000ULL,
      0x7ff8000000000001ULL,
      0x7ff0000000000001ULL,
  };
  for (uint64_t raw : inputs) {
    for (int mode = 0; mode < 4; ++mode) {
      uint64_t output = SENTINEL;
      EXPECT_FALSE(internal::exp64_candidate(raw, mode, &output));
      EXPECT_EQ(output, SENTINEL);
    }
  }
  uint64_t output = SENTINEL;
  EXPECT_FALSE(internal::exp64_candidate(FPBits(1.0).uintval(), -1, &output));
  EXPECT_EQ(output, SENTINEL);
  EXPECT_FALSE(internal::exp64_candidate(FPBits(1.0).uintval(), 4, &output));
  EXPECT_EQ(output, SENTINEL);
  EXPECT_FALSE(internal::exp64_candidate(FPBits(1.0).uintval(), 0, nullptr));
}

TEST_F(LlvmLibcExpTestIntegerEval, IntegerBackendBoundaries) {
  namespace internal = LIBC_NAMESPACE::math::exp_integer_internal;
  using UInt128 = internal::UInt128;
  UInt128 zero(0);
  UInt128 high = internal::one_at(127);
  EXPECT_EQ(internal::length(&zero), 0);
  EXPECT_EQ(internal::length(&high), 128);
  EXPECT_TRUE(internal::one_at(-1) == zero);
  EXPECT_TRUE(internal::one_at(128) == zero);
  EXPECT_TRUE(internal::shift(high, 128, 0) == zero);
  EXPECT_TRUE(internal::shift(high, -128, 0) == zero);
  EXPECT_TRUE(internal::shift(high, -128, 1) == UInt128(1));
  EXPECT_TRUE(internal::shift(high, -129, 1) == UInt128(1));
  EXPECT_TRUE(internal::shift(zero, -128, 1) == zero);
  EXPECT_TRUE(internal::shift(high, -127, 0) == UInt128(1));
  EXPECT_FALSE(internal::sticky(&high, 127));
  EXPECT_TRUE(internal::sticky(&high, 128));
  EXPECT_FALSE(internal::sticky(&zero, 129));
  EXPECT_EQ(internal::bit(&high, 127), unsigned(1));
  EXPECT_EQ(internal::bit(&high, 128), unsigned(0));
  EXPECT_EQ(internal::bit(&high, -1), unsigned(0));
  UInt128 bound = internal::one_at(65);
  UInt128 below = bound - UInt128(1);
  EXPECT_TRUE(internal::exceeds_bits(&bound, 65));
  EXPECT_FALSE(internal::exceeds_bits(&below, 65));
  bound = internal::one_at(57);
  below = bound - UInt128(1);
  EXPECT_TRUE(internal::exceeds_bits(&bound, 57));
  EXPECT_FALSE(internal::exceeds_bits(&below, 57));
  EXPECT_FALSE(internal::exceeds_bits(&high, 128));
  EXPECT_FALSE(internal::exceeds_bits(&zero, 0));
  EXPECT_TRUE(internal::exceeds_bits(&high, 0));
  EXPECT_TRUE(internal::exceeds_bits(&zero, -1));
}

TEST_F(LlvmLibcExpTestIntegerEval, IntegerBackendProducts) {
  namespace internal = LIBC_NAMESPACE::math::exp_integer_internal;
  using UInt128 = internal::UInt128;
  UInt128 a = internal::one_at(65) - UInt128(1);
  uint64_t b = (uint64_t(1) << 57) - 1;
  EXPECT_TRUE(internal::mul_q64(a, b, 0) == UInt128(0x03fffffffffffffdULL));
  EXPECT_TRUE(internal::mul_q64(a, b, 1) == UInt128(0x03fffffffffffffeULL));
  a = UInt128(b);
  EXPECT_TRUE(internal::mul_q64(a, UINT64_MAX, 0) ==
              UInt128(0x01fffffffffffffeULL));
  EXPECT_TRUE(internal::mul_q64(a, UINT64_MAX, 1) ==
              UInt128(0x01ffffffffffffffULL));
  EXPECT_TRUE(internal::mul_q64(internal::one_at(64), b, 0) == UInt128(b));
  EXPECT_TRUE(internal::mul_q64(internal::one_at(64), b, 1) == UInt128(b));
  EXPECT_TRUE(internal::mul_q64(UInt128(0), b, 1) == UInt128(0));
  EXPECT_TRUE(internal::mul_q64(UInt128(1), uint64_t(1), 1) == UInt128(1));
}

TEST_F(LlvmLibcExpTestIntegerEval, IntegerBackendRoundingIntervals) {
  namespace internal = LIBC_NAMESPACE::math::exp_integer_internal;
  using UInt128 = internal::UInt128;
  UInt128 half = internal::one_at(63), one = internal::one_at(64);
  UInt128 two = internal::one_at(65), midpoint = one + internal::one_at(11);
  const UInt128 pairs[][2] = {
      {half, half + UInt128(1)},
      {midpoint, midpoint},
      {midpoint - UInt128(1), midpoint + UInt128(1)},
      {one - UInt128(1), one + UInt128(1)},
      {two - UInt128(1), two + UInt128(1)},
      {half - UInt128(1), half + UInt128(1)},
      {internal::one_at(127), internal::one_at(127) + UInt128(1)},
  };
  const int scales[] = {-1100, -1023, -1022, -1, 0, 1, 1022, 1023, 1100};
  for (const auto &pair : pairs) {
    for (int q : scales) {
      for (int mode = 0; mode < 4; ++mode) {
        uint64_t lower = internal::round_binary64(pair[0], q, mode);
        uint64_t upper = internal::round_binary64(pair[1], q, mode);
        bool expected = lower != 0 && lower == upper;
        uint64_t output = UINT64_MAX;
        bool actual = internal::round_binary64_interval(pair[0], pair[1], q,
                                                        mode, &output);
        EXPECT_TRUE(actual == expected);
        EXPECT_EQ(output, expected ? lower : UINT64_MAX);
      }
    }
  }
}

TEST_F(LlvmLibcExpTestIntegerEval, IntegerBackendRoundingExpectedBits) {
  namespace internal = LIBC_NAMESPACE::math::exp_integer_internal;
  using UInt128 = internal::UInt128;
  const UInt128 one = internal::one_at(64);
  const UInt128 half_ulp = internal::one_at(11);
  struct Case {
    UInt128 lo;
    UInt128 hi;
    int scale;
    int mode;
    bool accepted;
    uint64_t expected;
  };
  const Case cases[] = {
      {one, one, 0, 0, true, 0x3ff0000000000000ULL},
      {one + half_ulp, one + half_ulp, 0, 0, true, 0x3ff0000000000000ULL},
      {one + UInt128(3) * half_ulp, one + UInt128(3) * half_ulp, 0, 0, true,
       0x3ff0000000000002ULL},
      {one + UInt128(1), one + UInt128(1), 0, 1, true, 0x3ff0000000000001ULL},
      {one + UInt128(1), one + UInt128(1), 0, 2, true, 0x3ff0000000000000ULL},
      {one + UInt128(1), one + UInt128(1), 0, 3, true, 0x3ff0000000000000ULL},
      {one + half_ulp - UInt128(1), one + half_ulp + UInt128(1), 0, 0, false,
       0},
      {internal::one_at(65) - UInt128(1), internal::one_at(65) - UInt128(1), 0,
       0, true, 0x4000000000000000ULL},
      {one, one, -1022, 0, true, 0x0010000000000000ULL},
      {one, one, -1023, 0, false, 0},
      {one, one, 1024, 0, false, 0},
      {internal::one_at(65) - UInt128(1), internal::one_at(65) - UInt128(1),
       1023, 0, false, 0},
      {internal::one_at(62), internal::one_at(62), 0, 0, true,
       0x3fd0000000000000ULL},
      {internal::one_at(127), internal::one_at(127), -63, 0, true,
       0x3ff0000000000000ULL},
  };
  for (const auto &test : cases) {
    uint64_t output = UINT64_MAX;
    EXPECT_EQ(internal::round_binary64_interval(test.lo, test.hi, test.scale,
                                                test.mode, &output),
              test.accepted);
    EXPECT_EQ(output, test.accepted ? test.expected : UINT64_MAX);
  }
}
