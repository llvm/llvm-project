//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains smoke tests for expf.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/math/expf_double_eval.h"
#include "src/__support/math/expf_float_eval.h"
#include "src/__support/math/expf_integer_eval.h"
#include "src/math/expf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class ExpfTest : public LIBC_NAMESPACE::testing::FPTest<float> {
public:
  void test_special_numbers(float (*func)(float),
                            bool check_snan_invalid = false,
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

    EXPECT_FP_EQ_ALL_ROUNDING(0.0f, func(neg_inf));
    if (check_errno)
      EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(1.0f, func(0.0f));
    if (check_errno)
      EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(1.0f, func(-0.0f));
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

#ifdef LIBC_TEST_FTZ_DAZ
  void test_denormals(float (*func)(float)) {
    EXPECT_FP_EQ(1.0f, func(min_denormal));
    EXPECT_FP_EQ(1.0f, func(max_denormal));
  }
#endif // LIBC_TEST_FTZ_DAZ
};

#ifdef LIBC_TEST_FTZ_DAZ
#define LIST_EXPF_FTZ_DAZ_TESTS(suffix, func)                                  \
  TEST_F(LlvmLibcExpfTest##suffix, FTZMode) {                                  \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ);  \
    test_denormals(&func);                                                     \
  }                                                                            \
  TEST_F(LlvmLibcExpfTest##suffix, DAZMode) {                                  \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::DAZ);  \
    test_denormals(&func);                                                     \
  }                                                                            \
  TEST_F(LlvmLibcExpfTest##suffix, FTZDAZMode) {                               \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ |  \
                                               LIBC_NAMESPACE::testing::DAZ);  \
    test_denormals(&func);                                                     \
  }
#else // !LIBC_TEST_FTZ_DAZ
#define LIST_EXPF_FTZ_DAZ_TESTS(suffix, func)
#endif // LIBC_TEST_FTZ_DAZ

#define LIST_EXPF_TESTS(suffix, func, check_snan_invalid,                      \
                        check_exception_and_errno, check_errno)                \
  using LlvmLibcExpfTest##suffix = ExpfTest;                                   \
  TEST_F(LlvmLibcExpfTest##suffix, SpecialNumbers) {                           \
    test_special_numbers(&func, check_snan_invalid, check_errno);              \
  }                                                                            \
  TEST_F(LlvmLibcExpfTest##suffix, Overflow) {                                 \
    test_overflow(&func, check_exception_and_errno);                           \
  }                                                                            \
  LIST_EXPF_FTZ_DAZ_TESTS(suffix, func)

LIST_EXPF_TESTS(Default, LIBC_NAMESPACE::expf, /*check_snan_invalid=*/true,
                /*check_exception_and_errno=*/true, /*check_errno=*/true)
LIST_EXPF_TESTS(DoubleEval, LIBC_NAMESPACE::math::double_eval::expf,
                /*check_snan_invalid=*/false,
                /*check_exception_and_errno=*/true, /*check_errno=*/true)
LIST_EXPF_TESTS(FloatEval, LIBC_NAMESPACE::math::float_eval::expf,
                /*check_snan_invalid=*/false,
                /*check_exception_and_errno=*/true, /*check_errno=*/true)
LIST_EXPF_TESTS(IntegerEval, LIBC_NAMESPACE::math::integer_eval::expf,
                /*check_snan_invalid=*/false,
                /*check_exception_and_errno=*/false, /*check_errno=*/false)

static float expf_static_rounding(float x) {
  return LIBC_NAMESPACE::shared::math::static_rounding::expf(
      x, LIBC_NAMESPACE::fputil::quick_get_round());
}

LIST_EXPF_TESTS(StaticRounding, expf_static_rounding,
                /*check_snan_invalid=*/false,
                /*check_exception_and_errno=*/false, /*check_errno=*/false)
