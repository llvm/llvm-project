//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains utility class to test different flavors of carg.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_COMPLEX_CARGTEST_H
#define LLVM_LIBC_TEST_SRC_COMPLEX_CARGTEST_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/math_macros.h"

template <typename CFPT, typename FPT>
class CArgTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(FPT)

public:
  using CArgFunc = FPT (*)(CFPT);

  void testZeroValues(CArgFunc func) {
    EXPECT_FP_EQ(func(CFPT{0.0, 0.0}), zero);
    EXPECT_FP_EQ(func(CFPT{0.0, -0.0}), neg_zero);
    EXPECT_FP_EQ(func(CFPT{1.0, 0.0}), zero);
    EXPECT_FP_EQ(func(CFPT{128.0, 0.0}), zero);
    EXPECT_FP_EQ(func(CFPT{1.0, -0.0}), neg_zero);
  }

  void testInfinityValues(CArgFunc func) {
    // carg(+inf + yi) = +0 for finite y > 0
    EXPECT_FP_EQ(zero, func(CFPT{inf, 1.0}));
    EXPECT_FP_EQ(zero, func(CFPT{inf, 256.0}));
    // carg(+inf - yi) = -0 for finite y > 0
    EXPECT_FP_EQ(neg_zero, func(CFPT{inf, -1.0}));
    EXPECT_FP_EQ(neg_zero, func(CFPT{inf, -1024.0}));
    // carg(+inf + 0i) = +0
    EXPECT_FP_EQ(zero, func(CFPT{inf, 0.0}));
    // carg(+inf - 0i) = -0
    EXPECT_FP_EQ(neg_zero, func(CFPT{inf, -0.0}));

    // carg(-inf + yi) = +pi for finite y > 0
    EXPECT_FP_EQ(FPT(M_PI), func(CFPT{neg_inf, 1.0}));
    EXPECT_FP_EQ(FPT(M_PI), func(CFPT{neg_inf, 64.0}));
    // carg(-inf - yi) = -pi for finite y > 0
    EXPECT_FP_EQ(FPT(-M_PI), func(CFPT{neg_inf, -1.0}));
    EXPECT_FP_EQ(FPT(-M_PI), func(CFPT{neg_inf, -512.0}));
    // carg(-inf + 0i) = +pi
    EXPECT_FP_EQ(FPT(M_PI), func(CFPT{neg_inf, 0.0}));
    // carg(-inf - 0i) = -pi
    EXPECT_FP_EQ(FPT(-M_PI), func(CFPT{neg_inf, -0.0}));

    // carg(x + inf*i) = +pi/2 for finite x
    EXPECT_FP_EQ(FPT(M_PI_2), func(CFPT{1.0, inf}));
    EXPECT_FP_EQ(FPT(M_PI_2), func(CFPT{-1.0, inf}));
    EXPECT_FP_EQ(FPT(M_PI_2), func(CFPT{0.0, inf}));
    EXPECT_FP_EQ(FPT(M_PI_2), func(CFPT{4.0, inf}));
    // carg(x - inf*i) = -pi/2 for finite x
    EXPECT_FP_EQ(FPT(-M_PI_2), func(CFPT{1.0, neg_inf}));
    EXPECT_FP_EQ(FPT(-M_PI_2), func(CFPT{-1.0, neg_inf}));
    EXPECT_FP_EQ(FPT(-M_PI_2), func(CFPT{0.0, neg_inf}));
    EXPECT_FP_EQ(FPT(-M_PI_2), func(CFPT{4.0, neg_inf}));
  }

  void testNaNValues(CArgFunc func) {
    // carg(NaN + yi) = NaN for finite y
    EXPECT_FP_IS_NAN(func(CFPT{aNaN, 0.0}));
    EXPECT_FP_IS_NAN(func(CFPT{aNaN, 1.0}));
    EXPECT_FP_IS_NAN(func(CFPT{aNaN, -1.0}));
    EXPECT_FP_IS_NAN(func(CFPT{aNaN, 512.0}));

    // carg(x + NaN*i) = NaN for finite x
    EXPECT_FP_IS_NAN(func(CFPT{0.0, aNaN}));
    EXPECT_FP_IS_NAN(func(CFPT{1.0, aNaN}));
    EXPECT_FP_IS_NAN(func(CFPT{-1.0, aNaN}));
    EXPECT_FP_IS_NAN(func(CFPT{4.0, aNaN}));

    // carg(NaN + NaN*i) = NaN
    EXPECT_FP_IS_NAN(func(CFPT{aNaN, aNaN}));

    // carg(+inf + NaN*i) = NaN
    EXPECT_FP_IS_NAN(func(CFPT{inf, aNaN}));
    // carg(-inf + NaN*i) = NaN
    EXPECT_FP_IS_NAN(func(CFPT{neg_inf, aNaN}));
    // carg(NaN + inf*i) = NaN
    EXPECT_FP_IS_NAN(func(CFPT{aNaN, inf}));
    // carg(NaN - inf*i) = NaN
    EXPECT_FP_IS_NAN(func(CFPT{aNaN, neg_inf}));
  }
};

#define LIST_CARG_TESTS(U, T, func)                                            \
  using LlvmLibcCArgTest = CArgTest<U, T>;                                     \
  TEST_F(LlvmLibcCArgTest, ZeroValues) { testZeroValues(&func); }              \
  TEST_F(LlvmLibcCArgTest, InfinityValues) { testInfinityValues(&func); }      \
  TEST_F(LlvmLibcCArgTest, NaNValues) { testNaNValues(&func); }

#endif // LLVM_LIBC_TEST_SRC_COMPLEX_CARGTEST_H
