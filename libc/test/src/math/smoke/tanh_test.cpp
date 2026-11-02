//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains smoke tests for tanh.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/tanh_double_eval.h"
#include "src/math/tanh.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

LIBC_NOINLINE double tanh_double_eval(double x) {
  return LIBC_NAMESPACE::math::double_eval::tanh(x);
}

class TanhTest : public LIBC_NAMESPACE::testing::FPTest<double> {
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

    EXPECT_FP_EQ_ALL_ROUNDING(1.0, func(inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(-1.0, func(neg_inf));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(0.0, func(0.0));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ALL_ROUNDING(-0.0, func(-0.0));
    EXPECT_MATH_ERRNO(0);
  }

  void test_saturation(double (*func)(double)) {
    // x = 0x1.fffffffffffffp+1023 (max_normal)
    EXPECT_FP_EQ_ALL_ROUNDING(
        1.0, 1.0, 0x1.fffffffffffffp-1, 0x1.fffffffffffffp-1,
        func(FPBits(uint64_t(0x7fef'ffff'ffff'ffff)).get_val()));
    EXPECT_MATH_ERRNO(0);

    // x = 0x1.30fc1931f09cap+4 (smallest positive input that saturates to 1.0
    // in FE_TONEAREST)
    EXPECT_FP_EQ_ALL_ROUNDING(
        1.0, 1.0, 0x1.fffffffffffffp-1, 0x1.fffffffffffffp-1,
        func(FPBits(uint64_t(0x4033'0fc1'931f'09ca)).get_val()));
    EXPECT_MATH_ERRNO(0);

    // x = -0x1.30fc1931f09cap+4 (largest negative input that saturates to -1.0
    // in FE_TONEAREST)
    EXPECT_FP_EQ_ALL_ROUNDING(
        -1.0, -0x1.fffffffffffffp-1, -1.0, -0x1.fffffffffffffp-1,
        func(FPBits(uint64_t(0xc033'0fc1'931f'09ca)).get_val()));
    EXPECT_MATH_ERRNO(0);
  }

  void test_underflow(double (*func)(double)) {
    EXPECT_FP_EQ_WITH_EXCEPTION(min_denormal, func(min_denormal), FE_UNDERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);

    EXPECT_FP_EQ_WITH_EXCEPTION(-min_denormal, func(-min_denormal),
                                FE_UNDERFLOW);
    EXPECT_MATH_ERRNO(ERANGE);
  }

  void test_accurate_pass(double (*func)(double)) {
#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
    // Hard-to-round inputs exercising tanh_small_accurate_128 and
    // tanh_accurate_128 without raising spurious exceptions:
    EXPECT_FP_EQ(0x1.fde1b6fff6ef9p-4, func(0x1.004524207fad9p-3));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ(-0x1.fde1b6fff6ef9p-4, func(-0x1.004524207fad9p-3));
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ(0x1.fefff82a29e4dp-5, func(0x1.ffaa08fcad74ep-5));
    EXPECT_MATH_ERRNO(0);
#else
    (void)func;
#endif // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  }

#ifdef LIBC_TEST_FTZ_DAZ
  void test_denormals(double (*func)(double)) {
    EXPECT_FP_EQ(0.0, func(min_denormal));
    EXPECT_FP_EQ(0.0, func(max_denormal));
  }
#endif // LIBC_TEST_FTZ_DAZ
};

#ifdef LIBC_TEST_FTZ_DAZ
#define LIST_TANH_FTZ_DAZ_TESTS(suffix, func)                                  \
  TEST_F(LlvmLibcTanhTest##suffix, FTZMode) {                                  \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ);  \
    test_denormals(&func);                                                     \
  }                                                                            \
  TEST_F(LlvmLibcTanhTest##suffix, DAZMode) {                                  \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::DAZ);  \
    test_denormals(&func);                                                     \
  }                                                                            \
  TEST_F(LlvmLibcTanhTest##suffix, FTZDAZMode) {                               \
    LIBC_NAMESPACE::testing::ModifyMXCSR mxcsr(LIBC_NAMESPACE::testing::FTZ |  \
                                               LIBC_NAMESPACE::testing::DAZ);  \
    test_denormals(&func);                                                     \
  }
#else // !LIBC_TEST_FTZ_DAZ
#define LIST_TANH_FTZ_DAZ_TESTS(suffix, func)
#endif // LIBC_TEST_FTZ_DAZ

#define LIST_TANH_TESTS(suffix, func, check_snan_invalid)                      \
  using LlvmLibcTanhTest##suffix = TanhTest;                                   \
  TEST_F(LlvmLibcTanhTest##suffix, SpecialNumbers) {                           \
    test_special_numbers(&func, check_snan_invalid);                           \
  }                                                                            \
  TEST_F(LlvmLibcTanhTest##suffix, Saturation) { test_saturation(&func); }     \
  TEST_F(LlvmLibcTanhTest##suffix, Underflow) { test_underflow(&func); }       \
  TEST_F(LlvmLibcTanhTest##suffix, AccuratePass) {                             \
    test_accurate_pass(&func);                                                 \
  }                                                                            \
  LIST_TANH_FTZ_DAZ_TESTS(suffix, func)                                        \
  static_assert(true, "Require semicolon.")

LIST_TANH_TESTS(Default, LIBC_NAMESPACE::tanh, /*check_snan_invalid=*/true);
LIST_TANH_TESTS(DoubleEval, tanh_double_eval, /*check_snan_invalid=*/false);
