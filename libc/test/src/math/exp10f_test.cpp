//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for exp10f.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/exp10f_double_eval.h"
#include "src/__support/math/exp10f_float_eval.h"
#include "src/math/exp10f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

class Exp10fTest : public LIBC_NAMESPACE::testing::FPTest<float> {
public:
  void test_special_numbers(float (*func)(float)) {
    EXPECT_FP_EQ(aNaN, func(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ(inf, func(inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ(0.0f, func(neg_inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ(1.0f, func(0.0f));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ(1.0f, func(-0.0f));
    EXPECT_MATH_ERRNO(0);
  }

  void test_overflow(float (*func)(float),
                     bool check_exception_and_errno = true) {
    constexpr float VALUES[] = {
        FPBits(0x7f7fffffU).get_val(),
        FPBits(0x43000000U).get_val(),
        FPBits(0x43000001U).get_val(),
    };
    for (float x : VALUES) {
      if (check_exception_and_errno) {
        EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(x), FE_OVERFLOW);
        EXPECT_MATH_ERRNO(ERANGE);
      } else {
        EXPECT_FP_EQ(inf, func(x));
      }
    }
  }

  void test_underflow(float (*func)(float), double ulp_tolerance = 0.5,
                      bool all_rounding = true,
                      bool check_exception_and_errno = true) {
    if (check_exception_and_errno) {
      EXPECT_FP_EQ_WITH_EXCEPTION(0.0f, func(FPBits(0xff7fffffU).get_val()),
                                  FE_UNDERFLOW);
      EXPECT_MATH_ERRNO(ERANGE);
    } else {
      EXPECT_FP_EQ(0.0f, func(FPBits(0xff7fffffU).get_val()));
    }

    constexpr float VALUES[] = {
        FPBits(0xc2cffff8U).get_val(),
        FPBits(0xc2d00008U).get_val(),
    };

    for (float x : VALUES) {
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp10, x, func(x), ulp_tolerance);
      }
      if (check_exception_and_errno) {
        EXPECT_MATH_ERRNO(ERANGE);
      }
    }
  }

  void test_tricky_inputs(float (*func)(float), double ulp_tolerance = 0.5,
                          bool all_rounding = true) {
    constexpr int N = 20;
    constexpr uint32_t INPUTS[N] = {
        0x325e5bd8, // x = 0x1.bcb7bp-27f
        0x325e5bd9, // x = 0x1.bcb7b2p-27f
        0x325e5bda, // x = 0x1.bcb7b4p-27f
        0x3d14d956, // x = 0x1.29b2acp-5f
        0x4116498a, // x = 0x1.2c9314p3f
        0x4126f431, // x = 0x1.4de862p3f
        0x4187d13c, // x = 0x1.0fa278p4f
        0x4203e9da, // x = 0x1.07d3b4p5f
        0x420b5f5d, // x = 0x1.16bebap5f
        0x42349e35, // x = 0x1.693c6ap5f
        0x3f800000, // x = 1.0f
        0x40000000, // x = 2.0f
        0x40400000, // x = 3.0f
        0x40800000, // x = 4.0f
        0x40a00000, // x = 5.0f
        0x40c00000, // x = 6.0f
        0x40e00000, // x = 7.0f
        0x41000000, // x = 8.0f
        0x41100000, // x = 9.0f
        0x41200000, // x = 10.0f
    };
    for (int i = 0; i < N; ++i) {
      float x = FPBits(INPUTS[i]).get_val();
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10, x, func(x),
                                       ulp_tolerance);
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10, -x, func(-x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp10, x, func(x), ulp_tolerance);
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp10, -x, func(-x), ulp_tolerance);
      }
    }
  }

  void test_in_range(float (*func)(float), double ulp_tolerance = 0.5,
                     bool all_rounding = true, bool check_errno = true) {
    constexpr uint32_t COUNT = 1'231;
    constexpr uint32_t STEP = UINT32_MAX / COUNT;
    for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      float x = FPBits(v).get_val();
      if (FPBits(v).is_nan() || FPBits(v).is_inf())
        continue;

      libc_errno = 0;
      float result = func(x);
      if (FPBits(result).is_nan() || FPBits(result).is_inf())
        continue;
      if (check_errno && libc_errno != 0)
        continue;

      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp10, x, result, ulp_tolerance);
      }
    }
  }
};

#define LIST_EXP10F_TESTS(suffix, func, ulp_tolerance, all_rounding)           \
  using LlvmLibcExp10fTest##suffix = Exp10fTest;                               \
  TEST_F(LlvmLibcExp10fTest##suffix, SpecialNumbers) {                         \
    test_special_numbers(&func);                                               \
  }                                                                            \
  TEST_F(LlvmLibcExp10fTest##suffix, Overflow) { test_overflow(&func); }       \
  TEST_F(LlvmLibcExp10fTest##suffix, Underflow) {                              \
    test_underflow(&func, ulp_tolerance, all_rounding);                        \
  }                                                                            \
  TEST_F(LlvmLibcExp10fTest##suffix, TrickyInputs) {                           \
    test_tricky_inputs(&func, ulp_tolerance, all_rounding);                    \
  }                                                                            \
  TEST_F(LlvmLibcExp10fTest##suffix, InFloatRange) {                           \
    test_in_range(&func, ulp_tolerance, all_rounding);                         \
  }

LIST_EXP10F_TESTS(Default, LIBC_NAMESPACE::exp10f, /*ulp_tolerance=*/0.5,
                  /*all_rounding=*/true)
LIST_EXP10F_TESTS(DoubleEval, LIBC_NAMESPACE::math::double_eval::exp10f,
                  /*ulp_tolerance=*/0.5, /*all_rounding=*/true)
LIST_EXP10F_TESTS(FloatEval, LIBC_NAMESPACE::math::float_eval::exp10f,
                  /*ulp_tolerance=*/1.5, /*all_rounding=*/false)
