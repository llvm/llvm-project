//===-- Utility class to test frexp[f|l] ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/BasicOperations.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T> class FrexpTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*FrexpFunc)(T, int *);

  void testSpecialNumbers(FrexpFunc func) {
    int exponent;
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(aNaN, &exponent));
    EXPECT_FP_EQ_ALL_ROUNDING(inf, func(inf, &exponent));
    EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, func(neg_inf, &exponent));

    EXPECT_FP_EQ_ALL_ROUNDING(0.0, func(0.0, &exponent));
    EXPECT_EQ(exponent, 0);

    EXPECT_FP_EQ_ALL_ROUNDING(-0.0, func(-0.0, &exponent));
    EXPECT_EQ(exponent, 0);
  }

  void testPowersOfTwo(FrexpFunc func) {
    int exponent;

    EXPECT_FP_EQ_ALL_ROUNDING(T(0.5), func(T(1.0), &exponent));
    EXPECT_EQ(exponent, 1);
    EXPECT_FP_EQ_ALL_ROUNDING(T(-0.5), func(T(-1.0), &exponent));
    EXPECT_EQ(exponent, 1);

    EXPECT_FP_EQ_ALL_ROUNDING(T(0.5), func(T(2.0), &exponent));
    EXPECT_EQ(exponent, 2);
    EXPECT_FP_EQ_ALL_ROUNDING(T(-0.5), func(T(-2.0), &exponent));
    EXPECT_EQ(exponent, 2);

    EXPECT_FP_EQ_ALL_ROUNDING(T(0.5), func(T(4.0), &exponent));
    EXPECT_EQ(exponent, 3);
    EXPECT_FP_EQ_ALL_ROUNDING(T(-0.5), func(T(-4.0), &exponent));
    EXPECT_EQ(exponent, 3);

    EXPECT_FP_EQ_ALL_ROUNDING(T(0.5), func(T(8.0), &exponent));
    EXPECT_EQ(exponent, 4);
    EXPECT_FP_EQ_ALL_ROUNDING(T(-0.5), func(T(-8.0), &exponent));
    EXPECT_EQ(exponent, 4);

    EXPECT_FP_EQ_ALL_ROUNDING(T(0.5), func(T(16.0), &exponent));
    EXPECT_EQ(exponent, 5);
    EXPECT_FP_EQ_ALL_ROUNDING(T(-0.5), func(T(-16.0), &exponent));
    EXPECT_EQ(exponent, 5);

    EXPECT_FP_EQ_ALL_ROUNDING(T(0.5), func(T(32.0), &exponent));
    EXPECT_EQ(exponent, 6);
    EXPECT_FP_EQ_ALL_ROUNDING(T(-0.5), func(T(-32.0), &exponent));
    EXPECT_EQ(exponent, 6);
  }

  void testSomeIntegers(FrexpFunc func) {
    int exponent;

    EXPECT_FP_EQ_ALL_ROUNDING(T(0.75), func(T(24.0), &exponent));
    EXPECT_EQ(exponent, 5);
    EXPECT_FP_EQ_ALL_ROUNDING(T(-0.75), func(T(-24.0), &exponent));
    EXPECT_EQ(exponent, 5);

    EXPECT_FP_EQ_ALL_ROUNDING(T(0.625), func(T(40.0), &exponent));
    EXPECT_EQ(exponent, 6);
    EXPECT_FP_EQ_ALL_ROUNDING(T(-0.625), func(T(-40.0), &exponent));
    EXPECT_EQ(exponent, 6);

    EXPECT_FP_EQ_ALL_ROUNDING(T(0.78125), func(T(800.0), &exponent));
    EXPECT_EQ(exponent, 10);
    EXPECT_FP_EQ_ALL_ROUNDING(T(-0.78125), func(T(-800.0), &exponent));
    EXPECT_EQ(exponent, 10);
  }
};

#define LIST_FREXP_TESTS(T, func)                                              \
  using LlvmLibcFrexpTest = FrexpTest<T>;                                      \
  TEST_F(LlvmLibcFrexpTest, SpecialNumbers) { testSpecialNumbers(&func); }     \
  TEST_F(LlvmLibcFrexpTest, PowersOfTwo) { testPowersOfTwo(&func); }           \
  TEST_F(LlvmLibcFrexpTest, SomeIntegers) { testSomeIntegers(&func); }         \
  static_assert(true, "Require semicolon.")
