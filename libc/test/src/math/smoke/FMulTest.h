//===-- Utility class to test fmul[f|l] ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMULTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMULTEST_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T, typename R>
class FmulTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*FMulFunc)(R, R);

  void testMul(FMulFunc func) {

    EXPECT_FP_EQ_ALL_ROUNDING(T(15.0), func(3.0, 5.0));
    EXPECT_FP_EQ_ALL_ROUNDING(T(0x1.0p-130), func(0x1.0p1, 0x1.0p-131));
    EXPECT_FP_EQ_ALL_ROUNDING(T(0x1.0p-127), func(0x1.0p2, 0x1.0p-129));
    EXPECT_FP_EQ_ALL_ROUNDING(T(1.0), func(1.0, 1.0));

    EXPECT_FP_EQ_ALL_ROUNDING(T(0.0), func(-0.0, -0.0));
    EXPECT_FP_EQ_ALL_ROUNDING(T(-0.0), func(0.0, -0.0));
    EXPECT_FP_EQ_ALL_ROUNDING(T(-0.0), func(-0.0, 0.0));

    EXPECT_FP_EQ_ROUNDING_NEAREST(inf, func(0x1.0p100, 0x1.0p100));
    EXPECT_FP_EQ_ROUNDING_UPWARD(inf, func(0x1.0p100, 0x1.0p100));
    EXPECT_FP_EQ_ROUNDING_DOWNWARD(max_normal, func(0x1.0p100, 0x1.0p100));
    EXPECT_FP_EQ_ROUNDING_TOWARD_ZERO(max_normal, func(0x1.0p100, 0x1.0p100));

    EXPECT_FP_EQ_ROUNDING_NEAREST(
        0x1p0, func(1.0, 1.0 + 0x1.0p-128 + 0x1.0p-149 + 0x1.0p-150));
    EXPECT_FP_EQ_ROUNDING_DOWNWARD(
        0x1p0, func(1.0, 1.0 + 0x1.0p-128 + 0x1.0p-149 + 0x1.0p-150));
    EXPECT_FP_EQ_ROUNDING_TOWARD_ZERO(
        0x1p0, func(1.0, 1.0 + 0x1.0p-128 + 0x1.0p-149 + 0x1.0p-150));
    EXPECT_FP_EQ_ROUNDING_UPWARD(
        0x1p0, func(1.0, 1.0 + 0x1.0p-128 + 0x1.0p-149 + 0x1.0p-150));

    EXPECT_FP_EQ_ROUNDING_NEAREST(
        0x1.0p-128f + 0x1.0p-148f,
        func(1.0, 0x1.0p-128 + 0x1.0p-149 + 0x1.0p-150));
    EXPECT_FP_EQ_ROUNDING_UPWARD(
        0x1.0p-128f + 0x1.0p-148f,
        func(1.0, 0x1.0p-128 + 0x1.0p-149 + 0x1.0p-150));
    EXPECT_FP_EQ_ROUNDING_DOWNWARD(
        0x1.0p-128f + 0x1.0p-149f,
        func(1.0, 0x1.0p-128 + 0x1.0p-149 + 0x1.0p-150));
    EXPECT_FP_EQ_ROUNDING_TOWARD_ZERO(
        0x1.0p-128f + 0x1.0p-149f,
        func(1.0, 0x1.0p-128 + 0x1.0p-149 + 0x1.0p-150));
  }

  void testSpecialInputs(FMulFunc func) {
    EXPECT_FP_EQ_ALL_ROUNDING(inf, func(inf, 0x1.0p-129));
    EXPECT_FP_EQ_ALL_ROUNDING(inf, func(0x1.0p-129, inf));
    EXPECT_FP_EQ_ALL_ROUNDING(inf, func(inf, 2.0));
    EXPECT_FP_EQ_ALL_ROUNDING(inf, func(3.0, inf));
    EXPECT_FP_EQ_ALL_ROUNDING(0.0, func(0.0, 0.0));

    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(neg_inf, aNaN));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(aNaN, neg_inf));
    EXPECT_FP_EQ_ALL_ROUNDING(inf, func(neg_inf, neg_inf));

    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(0.0, neg_inf));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(neg_inf, 0.0));

    EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, func(neg_inf, 1.0));
    EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, func(1.0, neg_inf));

    EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, func(neg_inf, 0x1.0p-129));
    EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, func(0x1.0p-129, neg_inf));

    EXPECT_FP_EQ_ALL_ROUNDING(0.0, func(0.0, 0x1.0p-129));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(inf, 0.0));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(0.0, inf));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(0.0, aNaN));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(2.0, aNaN));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(0x1.0p-129, aNaN));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(inf, aNaN));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(aNaN, aNaN));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(0.0, sNaN));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(2.0, sNaN));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(0x1.0p-129, sNaN));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(inf, sNaN));
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(sNaN, sNaN));
  }
};

#define LIST_FMUL_TESTS(T, R, func)                                            \
  using LlvmLibcFmulTest = FmulTest<T, R>;                                     \
  TEST_F(LlvmLibcFmulTest, Mul) { testMul(&func); }                            \
  TEST_F(LlvmLibcFmulTest, NaNInf) { testSpecialInputs(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMULTEST_H
