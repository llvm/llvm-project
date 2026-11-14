//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for log1pf.
///
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/log1pf_double_eval.h"
#include "src/math/log1pf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 1
#else
#define TOLERANCE 0
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

class Log1pfTest : public LIBC_NAMESPACE::testing::FPTest<float> {
public:
  void test_special_numbers(float (*func)(float), bool check_exception = true) {
    EXPECT_FP_EQ(aNaN, func(aNaN));
    EXPECT_FP_EQ(inf, func(inf));
    EXPECT_FP_EQ(zero, func(0.0f));
    EXPECT_FP_EQ(neg_zero, func(-0.0f));
    if (check_exception) {
      EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_inf), FE_INVALID);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(-1.0f), FE_DIVBYZERO);
    } else {
      EXPECT_FP_IS_NAN(func(neg_inf));
      EXPECT_FP_EQ(neg_inf, func(-1.0f));
    }
  }

  void test_tricky_inputs(float (*func)(float), double ulp_tolerance = 0.5,
                          bool all_rounding = true) {
    constexpr int N = 30;
    constexpr uint32_t INPUTS[N] = {
        0x35c00006U, /*0x1.80000cp-20f*/
        0x35400003U, /*0x1.800006p-21f*/
        0x3640000cU, /*0x1.800018p-19f*/
        0x36c00018U, /*0x1.80003p-18f*/
        0x3710001bU, /*0x1.200036p-17f*/
        0x37400030U, /*0x1.80006p-17f*/
        0x3770004bU, /*0x1.e00096p-17f*/
        0x3b9315c8U, /*0x1.262b9p-8f*/
        0x3c6eb7afU, /*0x1.dd6f5ep-7f*/
        0x3ddbfec3U, /*0x1.b7fd86p-4f*/
        0x3efd81adU, /*0x1.fb035ap-2f*/
        0x41078febU, /*0x1.0f1fd6p+3f*/
        0x4cc1c80bU, /*0x1.839016p+26f*/
        0x55185f82U, /*0x1.30bf04p+43f*/
        0x5cd69e88U, /*0x1.ad3d1p+58f*/
        0x5ee8984eU, /*0x1.d1309cp+62f*/
        0x65d890d3U, /*0x1.b121a6p+76f*/
        0x665e7ca6U, /*0x1.bcf94cp+77f*/
        0x6f31a8ecU, /*0x1.6351d8p+95f*/
        0x79e7ec37U, /*0x1.cfd86ep+116f*/
        0x7a17f30aU, /*0x1.2fe614p+117f*/
        0xb53ffffdU, /*-0x1.7ffffap-21f*/
        0xb5bffffaU, /*-0x1.7ffff4p-20f*/
        0xb63ffff4U, /*-0x1.7fffe8p-19f*/
        0xb70fffe5U, /*-0x1.1fffcap-17f*/
        0xbb0ec8c4U, /*-0x1.1d9188p-9f*/
        0xbc4d092cU, /*-0x1.9a1258p-7f*/
        0xbc657728U, /*-0x1.caee5p-7f*/
        0xbd1d20afU, /*-0x1.3a415ep-5f*/
        0xbf800000U, /*-1.0f*/
    };
    for (int i = 0; i < N; ++i) {
      float x = FPBits(INPUTS[i]).get_val();
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log1p, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Log1p, x, func(x), ulp_tolerance);
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
        ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log1p, x, func(x),
                                       ulp_tolerance);
      } else {
        ASSERT_MPFR_MATCH(mpfr::Operation::Log1p, x, func(x), ulp_tolerance);
      }
    }
  }
};

#define LIST_LOG1PF_TESTS(suffix, func, ulp_tolerance, all_rounding,           \
                          check_exception)                                     \
  using LlvmLibcLog1pfTest##suffix = Log1pfTest;                               \
  TEST_F(LlvmLibcLog1pfTest##suffix, SpecialNumbers) {                         \
    test_special_numbers(&func, check_exception);                              \
  }                                                                            \
  TEST_F(LlvmLibcLog1pfTest##suffix, TrickyInputs) {                           \
    test_tricky_inputs(&func, ulp_tolerance, all_rounding);                    \
  }                                                                            \
  TEST_F(LlvmLibcLog1pfTest##suffix, InFloatRange) {                           \
    test_in_range(&func, ulp_tolerance, all_rounding);                         \
  }

LIST_LOG1PF_TESTS(Default, LIBC_NAMESPACE::log1pf,
                  /*ulp_tolerance=*/TOLERANCE + 0.5, /*all_rounding=*/true,
                  /*check_exception=*/true)
LIST_LOG1PF_TESTS(DoubleEval, LIBC_NAMESPACE::math::double_eval::log1pf,
                  /*ulp_tolerance=*/TOLERANCE + 0.5, /*all_rounding=*/true,
                  /*check_exception=*/true)
