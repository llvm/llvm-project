//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for exp2f.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/exp2f_double_eval.h"
#include "src/__support/math/exp2f_float_eval.h"
#include "src/math/exp2f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 1
#else // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 0
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

class Exp2fTest : public LIBC_NAMESPACE::testing::FPTest<float> {
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

  void test_tricky_inputs(float (*func)(float), double ulp_tolerance = 0.5,
                          bool all_rounding = true) {
    constexpr int N = 12;
    constexpr uint32_t INPUTS[N] = {
        0x3b429d37U, /*0x1.853a6ep-9f*/
        0x3c02a9adU, /*0x1.05535ap-7f*/
        0x3ca66e26U, /*0x1.4cdc4cp-6f*/
        0x3d92a282U, /*0x1.254504p-4f*/
        0x42fa0001U, /*0x1.f40002p+6f*/
        0x42ffffffU, /*0x1.fffffep+6f*/
        0xb8d3d026U, /*-0x1.a7a04cp-14f*/
        0xbcf3a937U, /*-0x1.e7526ep-6f*/
        0xc2fa0001U, /*-0x1.f40002p+6f*/
        0xc2fc0000U, /*-0x1.f8p+6f*/
        0xc2fc0001U, /*-0x1.f80002p+6f*/
        0xc3150000U, /*-0x1.2ap+7f*/
    };
    for (int i = 0; i < N; ++i) {
      float x = FPBits(INPUTS[i]).get_val();
      libc_errno = 0;
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp2, x, func(x), ulp_tolerance);
      }
      EXPECT_MATH_ERRNO(0);
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

    struct TestCase {
      float x;
      int expected_errno;
    } cases[] = {
        {FPBits(0xc3158000U).get_val(), 0},
        {FPBits(0xc3160000U).get_val(), ERANGE},
        {FPBits(0xc3165432U).get_val(), ERANGE},
    };

    for (const auto &c : cases) {
      libc_errno = 0;
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, c.x, func(c.x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp2, c.x, func(c.x), ulp_tolerance);
      }
      if (check_exception_and_errno) {
        EXPECT_MATH_ERRNO(c.expected_errno);
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
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Exp2, x, result, ulp_tolerance);
      }
    }
  }
};

#define LIST_EXP2F_TESTS(suffix, func, ulp_tolerance, all_rounding)            \
  using LlvmLibcExp2fTest##suffix = Exp2fTest;                                 \
  TEST_F(LlvmLibcExp2fTest##suffix, SpecialNumbers) {                          \
    test_special_numbers(&func);                                               \
  }                                                                            \
  TEST_F(LlvmLibcExp2fTest##suffix, Overflow) { test_overflow(&func); }        \
  TEST_F(LlvmLibcExp2fTest##suffix, TrickyInputs) {                            \
    test_tricky_inputs(&func, ulp_tolerance, all_rounding);                    \
  }                                                                            \
  TEST_F(LlvmLibcExp2fTest##suffix, Underflow) {                               \
    test_underflow(&func, ulp_tolerance, all_rounding);                        \
  }                                                                            \
  TEST_F(LlvmLibcExp2fTest##suffix, InFloatRange) {                            \
    test_in_range(&func, ulp_tolerance, all_rounding);                         \
  }

LIST_EXP2F_TESTS(Default, LIBC_NAMESPACE::exp2f,
                 /*ulp_tolerance=*/TOLERANCE + 0.5, /*all_rounding=*/true)
LIST_EXP2F_TESTS(DoubleEval, LIBC_NAMESPACE::math::double_eval::exp2f,
                 /*ulp_tolerance=*/TOLERANCE + 0.5, /*all_rounding=*/true)
LIST_EXP2F_TESTS(FloatEval, LIBC_NAMESPACE::math::float_eval::exp2f,
                 /*ulp_tolerance=*/1.5, /*all_rounding=*/false)
