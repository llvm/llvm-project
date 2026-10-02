//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains utility class to test different flavors of cabs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_COMPLEX_CABSTEST_H
#define LLVM_LIBC_TEST_SRC_COMPLEX_CABSTEST_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/math_macros.h"

template <typename CFPT, typename FPT>
class CAbsTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(FPT)

public:
  using CAbsFunc = FPT (*)(CFPT);

  void testZeroValues(CAbsFunc func) {
    // cabs(+0 + 0i) = +0
    EXPECT_FP_EQ(zero, func(CFPT{0.0, 0.0}));
    // cabs(-0 + 0i) = +0
    EXPECT_FP_EQ(zero, func(CFPT{-0.0, 0.0}));
    // cabs(+0 - 0i) = +0
    EXPECT_FP_EQ(zero, func(CFPT{0.0, -0.0}));
    // cabs(-0 - 0i) = +0
    EXPECT_FP_EQ(zero, func(CFPT{-0.0, -0.0}));
  }

  void testBasicValues(CAbsFunc func) {
    // cabs(3 + 4i) = 5
    EXPECT_FP_EQ(FPT(5.0), func(CFPT{3.0, 4.0}));
    // cabs(-3 + 4i) = 5
    EXPECT_FP_EQ(FPT(5.0), func(CFPT{-3.0, 4.0}));
    // cabs(3 - 4i) = 5
    EXPECT_FP_EQ(FPT(5.0), func(CFPT{3.0, -4.0}));
    // cabs(-3 - 4i) = 5
    EXPECT_FP_EQ(FPT(5.0), func(CFPT{-3.0, -4.0}));
    // cabs(1 + 0i) = 1
    EXPECT_FP_EQ(FPT(1.0), func(CFPT{1.0, 0.0}));
    // cabs(0 + 1i) = 1
    EXPECT_FP_EQ(FPT(1.0), func(CFPT{0.0, 1.0}));
    // cabs(5 + 12i) = 13
    EXPECT_FP_EQ(FPT(13.0), func(CFPT{5.0, 12.0}));
  }

  void testInfinityValues(CAbsFunc func) {
    // cabs(+inf + yi) = +inf for finite y
    EXPECT_FP_EQ(inf, func(CFPT{inf, 0.0}));
    EXPECT_FP_EQ(inf, func(CFPT{inf, 1.0}));
    EXPECT_FP_EQ(inf, func(CFPT{inf, -1.0}));
    EXPECT_FP_EQ(inf, func(CFPT{inf, 256.0}));
    // cabs(-inf + yi) = +inf for finite y
    EXPECT_FP_EQ(inf, func(CFPT{neg_inf, 0.0}));
    EXPECT_FP_EQ(inf, func(CFPT{neg_inf, 1.0}));
    EXPECT_FP_EQ(inf, func(CFPT{neg_inf, -1.0}));
    EXPECT_FP_EQ(inf, func(CFPT{neg_inf, 512.0}));
    // cabs(x + inf*i) = +inf for finite x
    EXPECT_FP_EQ(inf, func(CFPT{0.0, inf}));
    EXPECT_FP_EQ(inf, func(CFPT{1.0, inf}));
    EXPECT_FP_EQ(inf, func(CFPT{-1.0, inf}));
    EXPECT_FP_EQ(inf, func(CFPT{4.0, inf}));
    // cabs(x - inf*i) = +inf for finite x
    EXPECT_FP_EQ(inf, func(CFPT{0.0, neg_inf}));
    EXPECT_FP_EQ(inf, func(CFPT{1.0, neg_inf}));
    EXPECT_FP_EQ(inf, func(CFPT{-1.0, neg_inf}));
    EXPECT_FP_EQ(inf, func(CFPT{4.0, neg_inf}));
    // cabs(+inf + inf*i) = +inf
    EXPECT_FP_EQ(inf, func(CFPT{inf, inf}));
    // cabs(-inf + inf*i) = +inf
    EXPECT_FP_EQ(inf, func(CFPT{neg_inf, inf}));
    // cabs(+inf + NaN*i) = +inf
    EXPECT_FP_EQ(inf, func(CFPT{inf, aNaN}));
    // cabs(-inf + NaN*i) = +inf
    EXPECT_FP_EQ(inf, func(CFPT{neg_inf, aNaN}));
    // cabs(NaN + inf*i) = +inf
    EXPECT_FP_EQ(inf, func(CFPT{aNaN, inf}));
    // cabs(NaN - inf*i) = +inf
    EXPECT_FP_EQ(inf, func(CFPT{aNaN, neg_inf}));
  }

  void testNaNValues(CAbsFunc func) {
    // cabs(NaN + yi) = NaN for finite y
    EXPECT_FP_IS_NAN(func(CFPT{aNaN, 0.0}));
    EXPECT_FP_IS_NAN(func(CFPT{aNaN, 1.0}));
    EXPECT_FP_IS_NAN(func(CFPT{aNaN, -1.0}));
    EXPECT_FP_IS_NAN(func(CFPT{aNaN, 512.0}));

    // cabs(x + NaN*i) = NaN for finite x
    EXPECT_FP_IS_NAN(func(CFPT{0.0, aNaN}));
    EXPECT_FP_IS_NAN(func(CFPT{1.0, aNaN}));
    EXPECT_FP_IS_NAN(func(CFPT{-1.0, aNaN}));
    EXPECT_FP_IS_NAN(func(CFPT{4.0, aNaN}));

    // cabs(NaN + NaN*i) = NaN
    EXPECT_FP_IS_NAN(func(CFPT{aNaN, aNaN}));
  }
};

#define LIST_CABS_TESTS(U, T, func)                                            \
  using LlvmLibcCAbsTest = CAbsTest<U, T>;                                     \
  TEST_F(LlvmLibcCAbsTest, ZeroValues) { testZeroValues(&func); }              \
  TEST_F(LlvmLibcCAbsTest, BasicValues) { testBasicValues(&func); }            \
  TEST_F(LlvmLibcCAbsTest, InfinityValues) { testInfinityValues(&func); }      \
  TEST_F(LlvmLibcCAbsTest, NaNValues) { testNaNValues(&func); }

#endif // LLVM_LIBC_TEST_SRC_COMPLEX_CABSTEST_H
