//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains smoke tests for exp10.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/math/exp10_double_eval.h"
#include "src/math/exp10.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

LIBC_NOINLINE double exp10_double_eval(double x) {
  return LIBC_NAMESPACE::math::double_eval::exp10(x);
}

class Exp10Test : public LIBC_NAMESPACE::testing::FPTest<double> {
public:
  void test_special_numbers(double (*func)(double),
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

    EXPECT_FP_EQ_ALL_ROUNDING(zero, func(neg_inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(1.0, func(0.0));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(1.0, func(-0.0));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(10.0, func(1.0));
    EXPECT_FP_EQ_ALL_ROUNDING(100.0, func(2.0));
    EXPECT_FP_EQ_ALL_ROUNDING(1000.0, func(3.0));
  }

  void test_overflow(double (*func)(double)) {
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(0x1.0p20), FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);
  }

  void test_underflow(double (*func)(double)) {
    EXPECT_FP_EQ_WITH_EXCEPTION(zero, func(-0x1.0p20), FE_UNDERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);
  }

#ifdef LIBC_TEST_FTZ_DAZ
  void test_denormals(double (*func)(double)) {
    EXPECT_FP_EQ(1.0, func(min_denormal));
    EXPECT_FP_EQ(1.0, func(max_denormal));
  }
#endif // LIBC_TEST_FTZ_DAZ
};

#ifdef LIBC_TEST_FTZ_DAZ
#define LIST_EXP10_FTZ_DAZ_TESTS(suffix, func)                                 \
  TEST_F(LlvmLibcExp10Test##suffix, FTZMode) {                                 \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ);  \
    test_denormals(&func);                                                     \
  }                                                                            \
  TEST_F(LlvmLibcExp10Test##suffix, DAZMode) {                                 \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::DAZ);  \
    test_denormals(&func);                                                     \
  }                                                                            \
  TEST_F(LlvmLibcExp10Test##suffix, FTZDAZMode) {                              \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ |  \
                                               LIBC_NAMESPACE::testing::DAZ);  \
    test_denormals(&func);                                                     \
  }
#else // !LIBC_TEST_FTZ_DAZ
#define LIST_EXP10_FTZ_DAZ_TESTS(suffix, func)
#endif // LIBC_TEST_FTZ_DAZ

#define LIST_EXP10_TESTS(suffix, func, check_snan_invalid)                     \
  using LlvmLibcExp10Test##suffix = Exp10Test;                                 \
  TEST_F(LlvmLibcExp10Test##suffix, SpecialNumbers) {                          \
    test_special_numbers(&func, check_snan_invalid);                           \
  }                                                                            \
  TEST_F(LlvmLibcExp10Test##suffix, Overflow) { test_overflow(&func); }        \
  TEST_F(LlvmLibcExp10Test##suffix, Underflow) { test_underflow(&func); }      \
  LIST_EXP10_FTZ_DAZ_TESTS(suffix, func)                                       \
  static_assert(true, "Require semicolon.")

LIST_EXP10_TESTS(Default, LIBC_NAMESPACE::exp10, /*check_snan_invalid=*/true);
LIST_EXP10_TESTS(DoubleEval, exp10_double_eval, /*check_snan_invalid=*/false);
