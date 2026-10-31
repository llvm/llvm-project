//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains smoke tests for sinh.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/math/sinh_double_eval.h"
#include "src/math/sinh.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

LIBC_NOINLINE double sinh_double_eval(double x) {
  return LIBC_NAMESPACE::math::double_eval::sinh(x);
}

class SinhTest : public LIBC_NAMESPACE::testing::FPTest<double> {
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

    EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, func(neg_inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(0.0, func(0.0));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(-0.0, func(-0.0));
    EXPECT_MATH_ERRNO(0);
  }

  void test_overflow(double (*func)(double)) {
    // x = 0x1.fffffffffffffp+1023 (max_normal)
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, func(FPBits(uint64_t(0x7fef'ffff'ffff'ffff)).get_val()),
        FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    // x = 0x1.633ce8fb9f87ep+9 (smallest positive input that overflows)
    EXPECT_FP_EQ_WITH_EXCEPTION(
        inf, func(FPBits(uint64_t(0x4086'33ce'8fb9'f87e)).get_val()),
        FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    // x = -0x1.633ce8fb9f87ep+9 (largest negative input that overflows)
    EXPECT_FP_EQ_WITH_EXCEPTION(
        neg_inf, func(FPBits(uint64_t(0xc086'33ce'8fb9'f87e)).get_val()),
        FE_OVERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);
  }

  void test_underflow(double (*func)(double)) {
    EXPECT_FP_EQ_WITH_EXCEPTION(min_denormal, func(min_denormal), FE_UNDERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION(-min_denormal, func(-min_denormal),
                                FE_UNDERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);
  }

#ifdef LIBC_TEST_FTZ_DAZ
  void test_denormals(double (*func)(double)) {
    EXPECT_FP_EQ(0.0, func(min_denormal));
    EXPECT_FP_EQ(0.0, func(max_denormal));
  }
#endif // LIBC_TEST_FTZ_DAZ
};

#ifdef LIBC_TEST_FTZ_DAZ
#define LIST_SINH_FTZ_DAZ_TESTS(suffix, func)                                  \
  TEST_F(LlvmLibcSinhTest##suffix, FTZMode) {                                  \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ);  \
    test_denormals(&func);                                                     \
  }                                                                            \
  TEST_F(LlvmLibcSinhTest##suffix, DAZMode) {                                  \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::DAZ);  \
    test_denormals(&func);                                                     \
  }                                                                            \
  TEST_F(LlvmLibcSinhTest##suffix, FTZDAZMode) {                               \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ |  \
                                               LIBC_NAMESPACE::testing::DAZ);  \
    test_denormals(&func);                                                     \
  }
#else // !LIBC_TEST_FTZ_DAZ
#define LIST_SINH_FTZ_DAZ_TESTS(suffix, func)
#endif // LIBC_TEST_FTZ_DAZ

#define LIST_SINH_TESTS(suffix, func, check_snan_invalid)                      \
  using LlvmLibcSinhTest##suffix = SinhTest;                                   \
  TEST_F(LlvmLibcSinhTest##suffix, SpecialNumbers) {                           \
    test_special_numbers(&func, check_snan_invalid);                           \
  }                                                                            \
  TEST_F(LlvmLibcSinhTest##suffix, Overflow) { test_overflow(&func); }         \
  TEST_F(LlvmLibcSinhTest##suffix, Underflow) { test_underflow(&func); }       \
  LIST_SINH_FTZ_DAZ_TESTS(suffix, func)                                        \
  static_assert(true, "Require semicolon.")

LIST_SINH_TESTS(Default, LIBC_NAMESPACE::sinh, /*check_snan_invalid=*/true);
LIST_SINH_TESTS(DoubleEval, sinh_double_eval, /*check_snan_invalid=*/false);
