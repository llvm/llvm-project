//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for log2f.
///
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/__support/math/log2f_double_eval.h"
#include "src/math/log2f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

class Log2fTest : public LIBC_NAMESPACE::testing::FPTest<float> {
public:
  void test_special_numbers(float (*func)(float), bool check_exception = true) {
    EXPECT_FP_EQ(aNaN, func(aNaN));
    EXPECT_FP_EQ(inf, func(inf));
    if (check_exception) {
      EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_inf), FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(0.0f), FE_DIVBYZERO);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(-0.0f), FE_DIVBYZERO);
      EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(-1.0f), FE_INVALID);
    } else {
      EXPECT_FP_IS_NAN(func(neg_inf));
      EXPECT_FP_EQ(neg_inf, func(0.0f));
      EXPECT_FP_EQ(neg_inf, func(-0.0f));
      EXPECT_FP_IS_NAN(func(-1.0f));
    }
    EXPECT_FP_EQ_ALL_ROUNDING(zero, func(1.0f));
  }

  void test_tricky_inputs(float (*func)(float), double ulp_tolerance = 0.5,
                          bool all_rounding = true) {
    constexpr int N = 12;
    constexpr uint32_t INPUTS[N] = {0x28114a90U, 0x3f7d57f5U, 0x3f7e3274U,
                                    0x3f7ed848U, 0x3f7fd6ccU, 0x3f7fffffU,
                                    0x3f80079bU, 0x3f81d0b5U, 0x3f82e602U,
                                    0x3f83c98dU, 0x3f8cba39U, 0x6e114a90U};

    for (int i = 0; i < N; ++i) {
      float x = FPBits(INPUTS[i]).get_val();
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log2, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Log2, x, func(x), ulp_tolerance);
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
      if (FPBits(result).is_nan() || FPBits(result).is_inf() ||
          (check_errno && libc_errno != 0))
        continue;
      if (all_rounding) {
        ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log2, x, func(x),
                                       ulp_tolerance);
      } else {
        ASSERT_MPFR_MATCH(mpfr::Operation::Log2, x, result, ulp_tolerance);
      }
    }
  }
};

#define LIST_LOG2F_TESTS(suffix, func, ulp_tolerance, all_rounding,            \
                         check_exception, check_errno)                         \
  using LlvmLibcLog2fTest##suffix = Log2fTest;                                 \
  TEST_F(LlvmLibcLog2fTest##suffix, SpecialNumbers) {                          \
    test_special_numbers(&func, check_exception);                              \
  }                                                                            \
  TEST_F(LlvmLibcLog2fTest##suffix, TrickyInputs) {                            \
    test_tricky_inputs(&func, ulp_tolerance, all_rounding);                    \
  }                                                                            \
  TEST_F(LlvmLibcLog2fTest##suffix, InFloatRange) {                            \
    test_in_range(&func, ulp_tolerance, all_rounding, check_errno);            \
  }

LIST_LOG2F_TESTS(Default, LIBC_NAMESPACE::log2f, /*ulp_tolerance=*/0.5,
                 /*all_rounding=*/true, /*check_exception=*/true,
                 /*check_errno=*/true)
LIST_LOG2F_TESTS(DoubleEval, LIBC_NAMESPACE::math::double_eval::log2f,
                 /*ulp_tolerance=*/0.5, /*all_rounding=*/true,
                 /*check_exception=*/true, /*check_errno=*/true)
