//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for expf.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/expf_double_eval.h"
#include "src/__support/math/expf_float_eval.h"
#include "src/__support/math/expf_integer_eval.h"
#include "src/math/expf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 1
#else // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 0
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

class ExpfTest : public LIBC_NAMESPACE::testing::FPTest<float> {
public:
  void test_special_numbers(float (*func)(float), bool check_errno = true) {
    EXPECT_FP_EQ(aNaN, func(aNaN));
    if (check_errno)
      EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ(inf, func(inf));
    if (check_errno)
      EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ(0.0f, func(neg_inf));
    if (check_errno)
      EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ(1.0f, func(0.0f));
    if (check_errno)
      EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ(1.0f, func(-0.0f));
    if (check_errno)
      EXPECT_MATH_ERRNO(0);
  }

  void test_overflow(float (*func)(float),
                     bool check_exception_and_errno = true) {
    constexpr float VALUES[] = {
        FPBits(0x7f7fffffU).get_val(),
        FPBits(0x42cffff8U).get_val(),
        FPBits(0x42d00008U).get_val(),
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
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp, x, func(x), ulp_tolerance);
      }
      if (check_exception_and_errno) {
        EXPECT_MATH_ERRNO(ERANGE);
      }
    }
  }

  void test_borderline(float (*func)(float), double ulp_tolerance = 0.5,
                       bool all_rounding = true, bool check_errno = true) {
    constexpr float INPUTS[] = {
        FPBits(0x42affff8U).get_val(), FPBits(0x42b00008U).get_val(),
        FPBits(0xc2affff8U).get_val(), FPBits(0xc2b00008U).get_val(),
        FPBits(0xc236bd8cU).get_val()};

    for (float x : INPUTS) {
      if (all_rounding) {
        ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, func(x),
                                       ulp_tolerance);
      } else {
        ASSERT_MPFR_MATCH(mpfr::Operation::Exp, x, func(x), ulp_tolerance);
      }
      if (check_errno) {
        EXPECT_MATH_ERRNO(0);
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
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp, x, result, ulp_tolerance);
      }
    }
  }
};

#define LIST_EXPF_TESTS(suffix, func, ulp_tolerance, all_rounding,             \
                        check_exception_and_errno, check_errno)                \
  using LlvmLibcExpfTest##suffix = ExpfTest;                                   \
  TEST_F(LlvmLibcExpfTest##suffix, SpecialNumbers) {                           \
    test_special_numbers(&func, check_errno);                                  \
  }                                                                            \
  TEST_F(LlvmLibcExpfTest##suffix, Overflow) {                                 \
    test_overflow(&func, check_exception_and_errno);                           \
  }                                                                            \
  TEST_F(LlvmLibcExpfTest##suffix, Underflow) {                                \
    test_underflow(&func, ulp_tolerance, all_rounding,                         \
                   check_exception_and_errno);                                 \
  }                                                                            \
  TEST_F(LlvmLibcExpfTest##suffix, Borderline) {                               \
    test_borderline(&func, ulp_tolerance, all_rounding, check_errno);          \
  }                                                                            \
  TEST_F(LlvmLibcExpfTest##suffix, InFloatRange) {                             \
    test_in_range(&func, ulp_tolerance, all_rounding, check_errno);            \
  }

LIST_EXPF_TESTS(Default, LIBC_NAMESPACE::expf,
                /*ulp_tolerance=*/TOLERANCE + 0.5,
                /*all_rounding=*/true, /*check_exception_and_errno=*/true,
                /*check_errno=*/true)
LIST_EXPF_TESTS(DoubleEval, LIBC_NAMESPACE::math::double_eval::expf,
                /*ulp_tolerance=*/TOLERANCE + 0.5, /*all_rounding=*/true,
                /*check_exception_and_errno=*/true, /*check_errno=*/true)
LIST_EXPF_TESTS(FloatEval, LIBC_NAMESPACE::math::float_eval::expf,
                /*ulp_tolerance=*/1.5, /*all_rounding=*/false,
                /*check_exception_and_errno=*/true, /*check_errno=*/true)
LIST_EXPF_TESTS(IntegerEval, LIBC_NAMESPACE::math::integer_eval::expf,
                /*ulp_tolerance=*/0.5, /*all_rounding=*/false,
                /*check_exception_and_errno=*/false, /*check_errno=*/false)

static float expf_static_rounding(float x) {
  return LIBC_NAMESPACE::shared::math::static_rounding::expf(
      x, LIBC_NAMESPACE::fputil::quick_get_round());
}

LIST_EXPF_TESTS(StaticRounding, expf_static_rounding,
                /*ulp_tolerance=*/0.5, /*all_rounding=*/true,
                /*check_exception_and_errno=*/false, /*check_errno=*/false)
