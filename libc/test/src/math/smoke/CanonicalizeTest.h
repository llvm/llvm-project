//===-- Utility class to test canonicalize[f|l] -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_CANONICALIZETEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_CANONICALIZETEST_H

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <math.h>

template <typename T>
class CanonicalizeTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*CanonicalizeFunc)(T);

  void testSpecialNumbers(CanonicalizeFunc func) {
    EXPECT_FP_EQ(zero, func(zero));
    EXPECT_FP_EQ(neg_zero, func(neg_zero));

    EXPECT_FP_EQ(inf, func(inf));
    EXPECT_FP_EQ(neg_inf, func(neg_inf));

    EXPECT_FP_EQ(aNaN, func(aNaN));
  }

  void testRoundedNumbers(CanonicalizeFunc func) {
    EXPECT_FP_EQ(T(1.0), func(T(1.0)));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.0)));
    EXPECT_FP_EQ(T(10.0), func(T(10.0)));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.0)));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.0)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.0)));
  }
};

#define LIST_CANONICALIZE_TESTS(T, func)                                       \
  using LlvmLibcCanonicalizeTest = CanonicalizeTest<T>;                        \
  TEST_F(LlvmLibcCanonicalizeTest, SpecialNumbers) {                           \
    testSpecialNumbers(&func);                                                 \
  }                                                                            \
  TEST_F(LlvmLibcCanonicalizeTest, RoundedNubmers) {                           \
    testRoundedNumbers(&func);                                                 \
  }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_CANONICALIZETEST_H
