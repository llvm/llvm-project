//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains smoke tests for log10f.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/math/log10f_double_eval.h"
#include "src/math/log10f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class Log10fTest : public LIBC_NAMESPACE::testing::FPTest<float> {
public:
  void test_special_numbers(float (*func)(float),
                            bool check_snan_invalid = false,
                            bool check_exception_and_errno = true,
                            bool check_errno = true) {
    if (check_snan_invalid) {
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN), FE_INVALID);
    } else {
      EXPECT_FP_EQ(aNaN, func(sNaN));
    }
    if (check_errno)
      EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(aNaN));
    if (check_errno)
      EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(inf, func(inf));
    if (check_errno)
      EXPECT_MATH_ERRNO(0);

    if (check_exception_and_errno) {
      EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_inf), FE_INVALID);
      EXPECT_MATH_ERRNO(EDOM);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(0.0f), FE_DIVBYZERO);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(-0.0f), FE_DIVBYZERO);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(-1.0f), FE_INVALID);
      EXPECT_MATH_ERRNO(EDOM);
    } else {
      EXPECT_FP_IS_NAN(func(neg_inf));
      EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, func(0.0f));
      EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, func(-0.0f));
      EXPECT_FP_IS_NAN(func(-1.0f));
    }

    EXPECT_FP_EQ_ALL_ROUNDING(zero, func(1.0f));

    float x = 1.0f;
    for (int i = 0; i < 11; ++i, x *= 10.0f) {
      EXPECT_FP_EQ_ALL_ROUNDING(static_cast<float>(i), func(x));
    }
  }

#ifdef LIBC_TEST_FTZ_DAZ
  void test_ftz(float (*func)(float)) {
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ);
    volatile float d = min_denormal;
    EXPECT_FP_EQ(-0x1.66d3e7bd9a403p5f, func(d));
  }

  void test_daz(float (*func)(float), bool uses_fpu) {
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::DAZ);
    volatile float d = min_denormal;
    EXPECT_FP_EQ(uses_fpu ? neg_inf : -0x1.66d3e7bd9a403p5f, func(d));
  }

  void test_ftzdaz(float (*func)(float), bool uses_fpu) {
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ |
                                               LIBC_NAMESPACE::testing::DAZ);
    volatile float d = min_denormal;
    EXPECT_FP_EQ(uses_fpu ? neg_inf : -0x1.66d3e7bd9a403p5f, func(d));
  }
#endif // LIBC_TEST_FTZ_DAZ
};

#ifdef LIBC_TEST_FTZ_DAZ
#define LIST_LOG10F_FTZ_DAZ_TESTS(suffix, func, uses_fpu)                      \
  TEST_F(LlvmLibcLog10fTest##suffix, FTZMode) { test_ftz(&func); }             \
  TEST_F(LlvmLibcLog10fTest##suffix, DAZMode) { test_daz(&func, uses_fpu); }   \
  TEST_F(LlvmLibcLog10fTest##suffix, FTZDAZMode) {                             \
    test_ftzdaz(&func, uses_fpu);                                              \
  }
#else // !LIBC_TEST_FTZ_DAZ
#define LIST_LOG10F_FTZ_DAZ_TESTS(suffix, func, uses_fpu)
#endif // LIBC_TEST_FTZ_DAZ

#define LIST_LOG10F_TESTS(suffix, func, check_snan_invalid,                    \
                          check_exception_and_errno, check_errno, uses_fpu)    \
  using LlvmLibcLog10fTest##suffix = Log10fTest;                               \
  TEST_F(LlvmLibcLog10fTest##suffix, SpecialNumbers) {                         \
    test_special_numbers(&func, check_snan_invalid, check_exception_and_errno, \
                         check_errno);                                         \
  }                                                                            \
  LIST_LOG10F_FTZ_DAZ_TESTS(suffix, func, uses_fpu)

LIST_LOG10F_TESTS(Default, LIBC_NAMESPACE::log10f, /*check_snan_invalid=*/true,
                  /*check_exception_and_errno=*/true, /*check_errno=*/true,
                  /*uses_fpu=*/true)
LIST_LOG10F_TESTS(DoubleEval, LIBC_NAMESPACE::math::double_eval::log10f,
                  /*check_snan_invalid=*/false,
                  /*check_exception_and_errno=*/true, /*check_errno=*/true,
                  /*uses_fpu=*/true)
