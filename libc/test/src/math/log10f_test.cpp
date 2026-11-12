//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for log10f.
///
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/log10f_double_eval.h"
#include "src/math/log10f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 1
#else
#define TOLERANCE 0
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

class Log10fTest : public LIBC_NAMESPACE::testing::FPTest<float> {
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
    constexpr int N = 22;
    constexpr uint32_t INPUTS[N] = {
        0x3f800000U /*1.0f*/,
        0x41200000U /*10.0f*/,
        0x42c80000U /*100.0f*/,
        0x447a0000U /*1,000.0f*/,
        0x461c4000U /*10,000.0f*/,
        0x47c35000U /*100,000.0f*/,
        0x49742400U /*1,000,000.0f*/,
        0x4b189680U /*10,000,000.0f*/,
        0x4cbebc20U /*100,000,000.0f*/,
        0x4e6e6b28U /*1,000,000,000.0f*/,
        0x501502f9U /*10,000,000,000.0f*/,
        0x4f134f83U /*2471461632.0f*/,
        0x610567e4U /*0x1.0acfc8p+67f*/,
        0x7956ba5eU /*69683218960000541503257137270226944.0f*/,
        0x08ae'a356U /*0x1.5d46acp-110f*/,
        0x1c7d'a337U /*0x1.fb466ep-71f*/,
        0x69c8'c583U /*0x1.918b06p+84f*/,
        0x0efe'ee7aU /*0x1.fddcf4p-98f*/,
        0x3f5f'de1bU /*0x1.bfbc36p-1f*/,
        0x3f80'70d8U /*0x1.00e1bp+0f*/,
        0x120b'93dcU /*0x1.1727b8p-91f*/,
        0x13ae'78d3U /*0x1.5cf1a6p-88f*/,
    };

    for (int i = 0; i < N; ++i) {
      float x = FPBits(INPUTS[i]).get_val();
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log10, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Log10, x, func(x), ulp_tolerance);
      }
    }
  }

  void test_in_range(float (*func)(float), double ulp_tolerance = 0.5,
                     bool all_rounding = true) {
    constexpr uint32_t COUNT = 1'231;
    constexpr uint32_t STEP = UINT32_MAX / COUNT;
    for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      float x = FPBits(v).get_val();
      if (FPBits(v).is_nan() || FPBits(v).is_inf())
        continue;
      if (all_rounding) {
        ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log10, x, func(x),
                                       ulp_tolerance);
      } else {
        ASSERT_MPFR_MATCH(mpfr::Operation::Log10, x, func(x), ulp_tolerance);
      }
    }
  }
};

#define LIST_LOG10F_TESTS(suffix, func, ulp_tolerance, all_rounding,           \
                          check_exception)                                     \
  using LlvmLibcLog10fTest##suffix = Log10fTest;                               \
  TEST_F(LlvmLibcLog10fTest##suffix, SpecialNumbers) {                         \
    test_special_numbers(&func, check_exception);                              \
  }                                                                            \
  TEST_F(LlvmLibcLog10fTest##suffix, TrickyInputs) {                           \
    test_tricky_inputs(&func, ulp_tolerance, all_rounding);                    \
  }                                                                            \
  TEST_F(LlvmLibcLog10fTest##suffix, InFloatRange) {                           \
    test_in_range(&func, ulp_tolerance, all_rounding);                         \
  }

LIST_LOG10F_TESTS(Default, LIBC_NAMESPACE::log10f,
                  /*ulp_tolerance=*/TOLERANCE + 0.5, /*all_rounding=*/true,
                  /*check_exception=*/true)
LIST_LOG10F_TESTS(DoubleEval, LIBC_NAMESPACE::math::double_eval::log10f,
                  /*ulp_tolerance=*/TOLERANCE + 0.5, /*all_rounding=*/true,
                  /*check_exception=*/true)
