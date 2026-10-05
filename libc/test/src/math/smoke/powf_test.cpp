//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for powf.
///
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/math/powf_double_eval.h"
#include "src/__support/math/powf_float_eval.h"
#include "src/math/powf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

class PowfTest : public LIBC_NAMESPACE::testing::FPTest<float> {
public:
  void test_special_numbers(float (*func)(float, float), int tolerance = 0) {
    constexpr float neg_odd_integer = -3.0f;
    constexpr float neg_even_integer = -6.0f;
    constexpr float neg_non_integer = -1.1f;
    constexpr float pos_odd_integer = 5.0f;
    constexpr float pos_even_integer = 8.0f;
    constexpr float pos_non_integer = 1.3f;
    constexpr float one_half = 0.5f;

    for (int i = 0; i < N_ROUNDING_MODES; ++i) {
      ForceRoundingMode __r(ROUNDING_MODES[i]);
      if (!__r.success)
        continue;

      // pow( sNaN, exponent)
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN, sNaN), FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN, neg_odd_integer),
                                  FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN, neg_even_integer),
                                  FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN, pos_odd_integer),
                                  FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN, pos_even_integer),
                                  FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN, one_half), FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN, zero), FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN, neg_zero), FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN, inf), FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN, neg_inf), FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN, aNaN), FE_INVALID);

      // pow( 0.0f, exponent )
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(zero, sNaN), FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(zero, neg_odd_integer),
                                  FE_DIVBYZERO);
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(zero, neg_even_integer),
                                  FE_DIVBYZERO);
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(zero, neg_non_integer),
                                  FE_DIVBYZERO);
      EXPECT_FP_EQ(zero, func(zero, pos_odd_integer));
      EXPECT_FP_EQ(zero, func(zero, pos_even_integer));
      EXPECT_FP_EQ(zero, func(zero, pos_non_integer));
      EXPECT_FP_EQ(zero, func(zero, one_half));
      EXPECT_FP_EQ(1.0f, func(zero, zero));
      EXPECT_FP_EQ(1.0f, func(zero, neg_zero));
      EXPECT_FP_EQ(0.0f, func(zero, inf));
      EXPECT_FP_EQ(inf, func(zero, neg_inf));
      EXPECT_FP_IS_NAN(func(zero, aNaN));

      // pow( -0.0f, exponent )
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(neg_zero, sNaN), FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(neg_zero, neg_odd_integer),
                                  FE_DIVBYZERO);
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(neg_zero, neg_even_integer),
                                  FE_DIVBYZERO);
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(neg_zero, neg_non_integer),
                                  FE_DIVBYZERO);
      EXPECT_FP_EQ(neg_zero, func(neg_zero, pos_odd_integer));
      EXPECT_FP_EQ(zero, func(neg_zero, pos_even_integer));
      EXPECT_FP_EQ(zero, func(neg_zero, pos_non_integer));
      EXPECT_FP_EQ(zero, func(neg_zero, one_half));
      EXPECT_FP_EQ(1.0f, func(neg_zero, zero));
      EXPECT_FP_EQ(1.0f, func(neg_zero, neg_zero));
      EXPECT_FP_EQ(0.0f, func(neg_zero, inf));
      EXPECT_FP_EQ(inf, func(neg_zero, neg_inf));
      EXPECT_FP_IS_NAN(func(neg_zero, aNaN));

      // pow( 1.0f, exponent )
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(1.0f, sNaN), FE_INVALID);
      EXPECT_FP_EQ(1.0f, func(1.0f, neg_odd_integer));
      EXPECT_FP_EQ(1.0f, func(1.0f, neg_even_integer));
      EXPECT_FP_EQ(1.0f, func(1.0f, neg_non_integer));
      EXPECT_FP_EQ(1.0f, func(1.0f, pos_odd_integer));
      EXPECT_FP_EQ(1.0f, func(1.0f, pos_even_integer));
      EXPECT_FP_EQ(1.0f, func(1.0f, pos_non_integer));
      EXPECT_FP_EQ(1.0f, func(1.0f, one_half));
      EXPECT_FP_EQ(1.0f, func(1.0f, zero));
      EXPECT_FP_EQ(1.0f, func(1.0f, neg_zero));
      EXPECT_FP_EQ(1.0f, func(1.0f, inf));
      EXPECT_FP_EQ(1.0f, func(1.0f, neg_inf));
      EXPECT_FP_EQ(1.0f, func(1.0f, aNaN));

      // pow( -1.0f, exponent )
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(-1.0f, sNaN), FE_INVALID);
      EXPECT_FP_EQ(-1.0f, func(-1.0f, neg_odd_integer));
      EXPECT_FP_EQ(1.0f, func(-1.0f, neg_even_integer));
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(-1.0f, neg_non_integer),
                                  FE_INVALID);
      EXPECT_FP_EQ(-1.0f, func(-1.0f, pos_odd_integer));
      EXPECT_FP_EQ(1.0f, func(-1.0f, pos_even_integer));
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(-1.0f, pos_non_integer),
                                  FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(-1.0f, one_half), FE_INVALID);
      EXPECT_FP_EQ(1.0f, func(-1.0f, zero));
      EXPECT_FP_EQ(1.0f, func(-1.0f, neg_zero));
      EXPECT_FP_EQ(1.0f, func(-1.0f, inf));
      EXPECT_FP_EQ(1.0f, func(-1.0f, neg_inf));
      EXPECT_FP_IS_NAN(func(-1.0f, aNaN));

      // pow( inf, exponent )
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(inf, sNaN), FE_INVALID);
      EXPECT_FP_EQ(0.0f, func(inf, neg_odd_integer));
      EXPECT_FP_EQ(0.0f, func(inf, neg_even_integer));
      EXPECT_FP_EQ(0.0f, func(inf, neg_non_integer));
      EXPECT_FP_EQ(inf, func(inf, pos_odd_integer));
      EXPECT_FP_EQ(inf, func(inf, pos_even_integer));
      EXPECT_FP_EQ(inf, func(inf, pos_non_integer));
      EXPECT_FP_EQ(inf, func(inf, one_half));
      EXPECT_FP_EQ(1.0f, func(inf, zero));
      EXPECT_FP_EQ(1.0f, func(inf, neg_zero));
      EXPECT_FP_EQ(inf, func(inf, inf));
      EXPECT_FP_EQ(0.0f, func(inf, neg_inf));
      EXPECT_FP_IS_NAN(func(inf, aNaN));

      // pow( -inf, exponent )
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(neg_inf, sNaN), FE_INVALID);
      EXPECT_FP_EQ(-0.0f, func(neg_inf, neg_odd_integer));
      EXPECT_FP_EQ(0.0f, func(neg_inf, neg_even_integer));
      EXPECT_FP_EQ(0.0f, func(neg_inf, neg_non_integer));
      EXPECT_FP_EQ(neg_inf, func(neg_inf, pos_odd_integer));
      EXPECT_FP_EQ(inf, func(neg_inf, pos_even_integer));
      EXPECT_FP_EQ(inf, func(neg_inf, pos_non_integer));
      EXPECT_FP_EQ(inf, func(neg_inf, one_half));
      EXPECT_FP_EQ(1.0f, func(neg_inf, zero));
      EXPECT_FP_EQ(1.0f, func(neg_inf, neg_zero));
      EXPECT_FP_EQ(inf, func(neg_inf, inf));
      EXPECT_FP_EQ(0.0f, func(neg_inf, neg_inf));
      EXPECT_FP_IS_NAN(func(neg_inf, aNaN));

      // pow( aNaN, exponent )
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(aNaN, sNaN), FE_INVALID);
      EXPECT_FP_IS_NAN(func(aNaN, neg_odd_integer));
      EXPECT_FP_IS_NAN(func(aNaN, neg_even_integer));
      EXPECT_FP_IS_NAN(func(aNaN, neg_non_integer));
      EXPECT_FP_IS_NAN(func(aNaN, pos_odd_integer));
      EXPECT_FP_IS_NAN(func(aNaN, pos_even_integer));
      EXPECT_FP_IS_NAN(func(aNaN, pos_non_integer));
      EXPECT_FP_IS_NAN(func(aNaN, one_half));
      EXPECT_FP_EQ(1.0f, func(aNaN, zero));
      EXPECT_FP_EQ(1.0f, func(aNaN, neg_zero));
      EXPECT_FP_IS_NAN(func(aNaN, inf));
      EXPECT_FP_IS_NAN(func(aNaN, neg_inf));
      EXPECT_FP_IS_NAN(func(aNaN, aNaN));

      // Exact powers of 2:
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(2.0f, sNaN), FE_INVALID);
      EXPECT_FP_EQ(0x1.0p15f, func(2.0f, 15.0f));
      EXPECT_FP_EQ(0x1.0p126f, func(2.0f, 126.0f));
      EXPECT_FP_EQ(0x1.0p-45f, func(2.0f, -45.0f));
      EXPECT_FP_EQ(0x1.0p-126f, func(2.0f, -126.0f));
      EXPECT_FP_EQ(0x1.0p-149f, func(2.0f, -149.0f));

      // Powers of 10:
      EXPECT_FP_EQ(1.0f, func(10.0f, 0.0f));
      EXPECT_FP_EQ(10.0f, func(10.0f, 1.0f));
      EXPECT_FP_EQ(100.0f, func(10.0f, 2.0f));
      if (tolerance == 0) {
        EXPECT_FP_EQ(1000.0f, func(10.0f, 3.0f));
        EXPECT_FP_EQ(10000.0f, func(10.0f, 4.0f));
        EXPECT_FP_EQ(100000.0f, func(10.0f, 5.0f));
        EXPECT_FP_EQ(1000000.0f, func(10.0f, 6.0f));
        EXPECT_FP_EQ(10000000.0f, func(10.0f, 7.0f));
        EXPECT_FP_EQ(100000000.0f, func(10.0f, 8.0f));
        EXPECT_FP_EQ(1000000000.0f, func(10.0f, 9.0f));
        EXPECT_FP_EQ(10000000000.0f, func(10.0f, 10.0f));
      } else {
        auto check_close = [tolerance](float act, float exp) {
          uint32_t a = FPBits(act).uintval();
          uint32_t e = FPBits(exp).uintval();
          EXPECT_LE(a >= e ? a - e : e - a, static_cast<uint32_t>(tolerance));
        };
        check_close(func(10.0f, 3.0f), 1000.0f);
        check_close(func(10.0f, 4.0f), 10000.0f);
        check_close(func(10.0f, 5.0f), 100000.0f);
        check_close(func(10.0f, 6.0f), 1000000.0f);
        check_close(func(10.0f, 7.0f), 10000000.0f);
        check_close(func(10.0f, 8.0f), 100000000.0f);
        check_close(func(10.0f, 9.0f), 1000000000.0f);
        check_close(func(10.0f, 10.0f), 10000000000.0f);
      }
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(10.0f, sNaN), FE_INVALID);

      // Overflow / Underflow:
      if (ROUNDING_MODES[i] != RoundingMode::Downward &&
          ROUNDING_MODES[i] != RoundingMode::TowardZero) {
        EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(3.1f, 201.0f), FE_OVERFLOW);
      }
      if (ROUNDING_MODES[i] != RoundingMode::Upward) {
        EXPECT_FP_EQ_WITH_EXCEPTION(0.0f, func(3.1f, -201.0f), FE_UNDERFLOW);
      }
    }

    EXPECT_FP_EQ(-0.0f, func(-0.015625f, 25.0f));
    EXPECT_FP_EQ(0.0f, func(-0.015625f, 26.0f));
  }

  void test_subnormal_base(float (*func)(float, float), int tolerance = 0) {
    EXPECT_FP_EQ(0x1.0p-32f, func(0x1.0p-128f, 0.25f));
    EXPECT_FP_EQ(0x1.0p96f, func(0x1.0p-128f, -0.75f));
    if (tolerance == 0) {
      EXPECT_FP_EQ(0x1.90a962p-33f, func(0x1.8p-130f, 0.25f));
      EXPECT_FP_EQ(0x1.47238cp+32f, func(0x1.8p-130f, -0.25f));
    } else {
      uint32_t act1 = FPBits(func(0x1.8p-130f, 0.25f)).uintval();
      uint32_t exp1 = FPBits(0x1.90a962p-33f).uintval();
      EXPECT_LE(act1 >= exp1 ? act1 - exp1 : exp1 - act1,
                static_cast<uint32_t>(tolerance));
      uint32_t act2 = FPBits(func(0x1.8p-130f, -0.25f)).uintval();
      uint32_t exp2 = FPBits(0x1.47238cp+32f).uintval();
      EXPECT_LE(act2 >= exp2 ? act2 - exp2 : exp2 - act2,
                static_cast<uint32_t>(tolerance));
    }
  }

#ifdef LIBC_TEST_FTZ_DAZ
  void test_ftz(float (*func)(float, float)) {
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ);
    volatile float x = -min_denormal;
    volatile float y = 0.5f;
    EXPECT_FP_IS_NAN(func(x, y));
    volatile float two = 2.0f;
    volatile float d = min_denormal;
    EXPECT_FP_EQ(1.0f, func(two, d));
  }

  void test_daz(float (*func)(float, float)) {
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::DAZ);
    volatile float x = -min_denormal;
    volatile float y = 0.5f;
    EXPECT_FP_EQ(0.0f, func(x, y));
    volatile float two = 2.0f;
    volatile float d = min_denormal;
    EXPECT_FP_EQ(1.0f, func(two, d));
  }

  void test_ftzdaz(float (*func)(float, float)) {
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ |
                                               LIBC_NAMESPACE::testing::DAZ);
    volatile float x = -min_denormal;
    volatile float y = 0.5f;
    EXPECT_FP_EQ(0.0f, func(x, y));
    volatile float two = 2.0f;
    volatile float d = min_denormal;
    EXPECT_FP_EQ(1.0f, func(two, d));
  }
#endif // LIBC_TEST_FTZ_DAZ
};

#ifdef LIBC_TEST_FTZ_DAZ
#define LIST_POWF_FTZ_DAZ_TESTS(suffix, func)                                  \
  TEST_F(LlvmLibcPowfTest##suffix, FTZMode) { test_ftz(&func); }               \
  TEST_F(LlvmLibcPowfTest##suffix, DAZMode) { test_daz(&func); }               \
  TEST_F(LlvmLibcPowfTest##suffix, FTZDAZMode) { test_ftzdaz(&func); }
#else
#define LIST_POWF_FTZ_DAZ_TESTS(suffix, func)
#endif // LIBC_TEST_FTZ_DAZ

#define LIST_POWF_TESTS(suffix, func, tolerance)                               \
  using LlvmLibcPowfTest##suffix = PowfTest;                                   \
  TEST_F(LlvmLibcPowfTest##suffix, SpecialNumbers) {                           \
    test_special_numbers(&func, tolerance);                                    \
  }                                                                            \
  TEST_F(LlvmLibcPowfTest##suffix, SubnormalBase) {                            \
    test_subnormal_base(&func, tolerance);                                     \
  }                                                                            \
  LIST_POWF_FTZ_DAZ_TESTS(suffix, func)                                        \
  static_assert(true, "Require semicolon.")

LIST_POWF_TESTS(Default, LIBC_NAMESPACE::powf, /*tolerance=*/0);
LIST_POWF_TESTS(DoubleEval, LIBC_NAMESPACE::math::double_eval::powf,
                /*tolerance=*/0);
LIST_POWF_TESTS(FloatEval, LIBC_NAMESPACE::math::float_eval::powf,
                /*tolerance=*/1);
