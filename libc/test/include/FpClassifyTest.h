//===-- Utility class to test the fpclassify macro  -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_INCLUDE_MATH_FPCLASSIFY_H
#define LLVM_LIBC_TEST_INCLUDE_MATH_FPCLASSIFY_H

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/math-function-macros.h"

template <typename T>
class FpClassifyTest : public LIBC_NAMESPACE::testing::Test {
  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef int (*FpClassifyFunc)(T, T, T, T, T, T);

  void testSpecialNumbers(FpClassifyFunc func) {
    EXPECT_EQ(func(1, 2, 3, 4, 5, aNaN), 1);
    EXPECT_EQ(func(1, 2, 3, 4, 5, neg_aNaN), 1);
    EXPECT_EQ(func(1, 2, 3, 4, 5, sNaN), 1);
    EXPECT_EQ(func(1, 2, 3, 4, 5, neg_sNaN), 1);
    EXPECT_EQ(func(1, 2, 3, 4, 5, inf), 2);
    EXPECT_EQ(func(1, 2, 3, 4, 5, neg_inf), 2);
    EXPECT_EQ(func(1, 2, 3, 4, 5, min_normal), 3);
    EXPECT_EQ(func(1, 2, 3, 4, 5, max_normal), 3);
    EXPECT_EQ(func(1, 2, 3, 4, 5, neg_max_normal), 3);
    EXPECT_EQ(func(1, 2, 3, 4, 5, min_denormal), 4);
    EXPECT_EQ(func(1, 2, 3, 4, 5, neg_min_denormal), 4);
    EXPECT_EQ(func(1, 2, 3, 4, 5, max_denormal), 4);
    EXPECT_EQ(func(1, 2, 3, 4, 5, zero), 5);
    EXPECT_EQ(func(1, 2, 3, 4, 5, neg_zero), 5);
  }
};

#define LIST_FPCLASSIFY_TESTS(T, func)                                         \
  using LlvmLibcFpClassifyTest = FpClassifyTest<T>;                            \
  TEST_F(LlvmLibcFpClassifyTest, SpecialNumbers) {                             \
    auto fpclassify_func = [](T a, T b, T c, T d, T e, T f) {                  \
      return func(a, b, c, d, e, f);                                           \
    };                                                                         \
    testSpecialNumbers(fpclassify_func);                                       \
  }

#endif // LLVM_LIBC_TEST_INCLUDE_MATH_FPCLASSIFY_H
