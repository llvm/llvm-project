//===-- Utility class to test ceil[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <math.h>

template <typename T> class CeilTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*CeilFunc)(T);

  void testSpecialNumbers(CeilFunc func) {
    EXPECT_FP_EQ(zero, func(zero));
    EXPECT_FP_EQ(neg_zero, func(neg_zero));

    EXPECT_FP_EQ(inf, func(inf));
    EXPECT_FP_EQ(neg_inf, func(neg_inf));

    EXPECT_FP_EQ(aNaN, func(aNaN));
  }

  void testRoundedNumbers(CeilFunc func) {
    EXPECT_FP_EQ(T(1.0), func(T(1.0)));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.0)));
    EXPECT_FP_EQ(T(10.0), func(T(10.0)));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.0)));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.0)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.0)));
  }

  void testFractions(CeilFunc func) {
    EXPECT_FP_EQ(T(1.0), func(T(0.5)));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.5)));
    EXPECT_FP_EQ(T(1.0), func(T(0.115)));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.115)));
    EXPECT_FP_EQ(T(1.0), func(T(0.715)));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.715)));
    EXPECT_FP_EQ(T(2.0), func(T(1.3)));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.3)));
    EXPECT_FP_EQ(T(2.0), func(T(1.5)));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.5)));
    EXPECT_FP_EQ(T(2.0), func(T(1.75)));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.75)));
    EXPECT_FP_EQ(T(11.0), func(T(10.32)));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.32)));
    EXPECT_FP_EQ(T(11.0), func(T(10.65)));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.65)));
    EXPECT_FP_EQ(T(1235.0), func(T(1234.38)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.38)));
    EXPECT_FP_EQ(T(1235.0), func(T(1234.96)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.96)));
  }
};

#define LIST_CEIL_TESTS(T, func)                                               \
  using LlvmLibcCeilTest = CeilTest<T>;                                        \
  TEST_F(LlvmLibcCeilTest, SpecialNumbers) { testSpecialNumbers(&func); }      \
  TEST_F(LlvmLibcCeilTest, RoundedNubmers) { testRoundedNumbers(&func); }      \
  TEST_F(LlvmLibcCeilTest, Fractions) { testFractions(&func); }
