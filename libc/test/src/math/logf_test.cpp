//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittests for logf.
///
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/logf_double_eval.h"
#include "src/math/logf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 1
#else
#define TOLERANCE 0
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

class LogfTest : public LIBC_NAMESPACE::testing::FPTest<float> {
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
#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
    EXPECT_TRUE(0.0f == func(1.0f));
#else
    EXPECT_FP_EQ_ALL_ROUNDING(zero, func(1.0f));
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  }

  void test_tricky_inputs(float (*func)(float), double ulp_tolerance = 0.5,
                          bool all_rounding = true) {
    constexpr int N = 35;
    constexpr uint32_t INPUTS[N] = {
        0x1b7679ffU, /*0x1.ecf3fep-73f*/
        0x1e88452dU, /*0x1.108a5ap-66f*/
        0x3f800001U, /*0x1.000002p+0f*/
        0x3509dcf6U, /*0x1.13b9ecp-21f*/
        0x3bf86ef0U, /*0x1.f0ddep-8f*/
        0x3c413d3aU, /*0x1.827a74p-7f*/
        0x3ca1c99fU, /*0x1.43933ep-6f*/
        0x3d13e105U, /*0x1.27c20ap-5f*/
        0x3f7ff1f2U, /*0x1.ffe3e4p-1f*/
        0x3f7fffffU, /*0x1.fffffep-1f*/
        0x3f800001U, /*0x1.000002p+0f*/
        0x3f800006U, /*0x1.00000cp+0f*/
        0x3f800014U, /*0x1.000028p+0f*/
        0x3f80001cU, /*0x1.000038p+0f*/
        0x3f80c777U, /*0x1.018eeep+0f*/
        0x3f80ce72U, /*0x1.019ce4p+0f*/
        0x3f80d19fU, /*0x1.01a33ep+0f*/
        0x3f80f7bfU, /*0x1.01ef7ep+0f*/
        0x3f80fcfeU, /*0x1.01f9fcp+0f*/
        0x3f81feb4U, /*0x1.03fd68p+0f*/
        0x3f83d731U, /*0x1.07ae62p+0f*/
        0x3f90cb1dU, /*0x1.21963ap+0f*/
        0x3fc55379U, /*0x1.8aa6f2p+0f*/
        0x3fd364d7U, /*0x1.a6c9aep+0f*/
        0x41178febU, /*0x1.2f1fd6p+3f*/
        0x4c5d65a5U, /*0x1.bacb4ap+25f*/
        0x4e85f412U, /*0x1.0be824p+30f*/
        0x500ffb03U, /*0x1.1ff606p+33f*/
        0x5cd69e88U, /*0x1.ad3d1p+58f*/
        0x5ee8984eU, /*0x1.d1309cp+62f*/
        0x65d890d3U, /*0x1.b121a6p+76f*/
        0x665e7ca6U, /*0x1.bcf94cp+77f*/
        0x6f31a8ecU, /*0x1.6351d8p+95f*/
        0x79e7ec37U, /*0x1.cfd86ep+116f*/
        0x7a17f30aU, /*0x1.2fe614p+117f*/
    };
    for (int i = 0; i < N; ++i) {
      float x = FPBits(INPUTS[i]).get_val();
      if (all_rounding) {
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log, x, func(x),
                                       ulp_tolerance);
      } else {
        EXPECT_MPFR_MATCH(mpfr::Operation::Log, x, func(x), ulp_tolerance);
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
        ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log, x, func(x),
                                       ulp_tolerance);
      } else {
        ASSERT_MPFR_MATCH(mpfr::Operation::Log, x, func(x), ulp_tolerance);
      }
    }
  }
};

#define LIST_LOGF_TESTS(suffix, func, ulp_tolerance, all_rounding,             \
                        check_exception)                                       \
  using LlvmLibcLogfTest##suffix = LogfTest;                                   \
  TEST_F(LlvmLibcLogfTest##suffix, SpecialNumbers) {                           \
    test_special_numbers(&func, check_exception);                              \
  }                                                                            \
  TEST_F(LlvmLibcLogfTest##suffix, TrickyInputs) {                             \
    test_tricky_inputs(&func, ulp_tolerance, all_rounding);                    \
  }                                                                            \
  TEST_F(LlvmLibcLogfTest##suffix, InFloatRange) {                             \
    test_in_range(&func, ulp_tolerance, all_rounding);                         \
  }

LIST_LOGF_TESTS(Default, LIBC_NAMESPACE::logf,
                /*ulp_tolerance=*/TOLERANCE + 0.5, /*all_rounding=*/true,
                /*check_exception=*/true)
LIST_LOGF_TESTS(DoubleEval, LIBC_NAMESPACE::math::double_eval::logf,
                /*ulp_tolerance=*/TOLERANCE + 0.5, /*all_rounding=*/true,
                /*check_exception=*/true)
