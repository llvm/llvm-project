//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains smoke tests for exp2f.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/math/exp2f_double_eval.h"
#include "src/__support/math/exp2f_float_eval.h"
#include "src/math/exp2f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class Exp2fTest : public LIBC_NAMESPACE::testing::FPTest<float> {
public:
  void test_special_numbers(float (*func)(float),
                            bool check_snan_invalid = false) {
    if (check_snan_invalid) {
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(sNaN), FE_INVALID);
    } else {
      EXPECT_FP_EQ(aNaN, func(sNaN));
    }
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(aNaN));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(inf, func(inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(0.0f, func(neg_inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(1.0f, func(0.0f));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(1.0f, func(-0.0f));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(2.0f, func(1.0f));
    EXPECT_FP_EQ_ALL_ROUNDING(0.5f, func(-1.0f));
    EXPECT_FP_EQ_ALL_ROUNDING(4.0f, func(2.0f));
    EXPECT_FP_EQ_ALL_ROUNDING(0.25f, func(-2.0f));
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

#ifdef LIBC_TEST_FTZ_DAZ
  void test_denormals(float (*func)(float)) {
    EXPECT_FP_EQ(1.0f, func(min_denormal));
    EXPECT_FP_EQ(1.0f, func(max_denormal));
  }
#endif // LIBC_TEST_FTZ_DAZ
};

#ifdef LIBC_TEST_FTZ_DAZ
#define LIST_EXP2F_FTZ_DAZ_TESTS(suffix, func)                                 \
  TEST_F(LlvmLibcExp2fTest##suffix, FTZMode) {                                 \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ);  \
    test_denormals(&func);                                                     \
  }                                                                            \
  TEST_F(LlvmLibcExp2fTest##suffix, DAZMode) {                                 \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::DAZ);  \
    test_denormals(&func);                                                     \
  }                                                                            \
  TEST_F(LlvmLibcExp2fTest##suffix, FTZDAZMode) {                              \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ |  \
                                               LIBC_NAMESPACE::testing::DAZ);  \
    test_denormals(&func);                                                     \
  }
#else // !LIBC_TEST_FTZ_DAZ
#define LIST_EXP2F_FTZ_DAZ_TESTS(suffix, func)
#endif // LIBC_TEST_FTZ_DAZ

#define LIST_EXP2F_TESTS(suffix, func, check_snan_invalid)                     \
  using LlvmLibcExp2fTest##suffix = Exp2fTest;                                 \
  TEST_F(LlvmLibcExp2fTest##suffix, SpecialNumbers) {                          \
    test_special_numbers(&func, check_snan_invalid);                           \
  }                                                                            \
  TEST_F(LlvmLibcExp2fTest##suffix, Overflow) { test_overflow(&func); }        \
  LIST_EXP2F_FTZ_DAZ_TESTS(suffix, func)

LIST_EXP2F_TESTS(Default, LIBC_NAMESPACE::exp2f, /*check_snan_invalid=*/true)
LIST_EXP2F_TESTS(DoubleEval, LIBC_NAMESPACE::math::double_eval::exp2f,
                 /*check_snan_invalid=*/false)
LIST_EXP2F_TESTS(FloatEval, LIBC_NAMESPACE::math::float_eval::exp2f,
                 /*check_snan_invalid=*/false)
