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
  typedef int (*FpClassifyFunc)(T);

  void testSpecialNumbers(FpClassifyFunc func) {
    EXPECT_EQ(func(aNaN), 0);
    EXPECT_EQ(func(neg_aNaN), 0);
    EXPECT_EQ(func(sNaN), 0);
    EXPECT_EQ(func(neg_sNaN), 0);
    EXPECT_EQ(func(inf), 1);
    EXPECT_EQ(func(neg_inf), 1);
    EXPECT_EQ(func(min_normal), 4);
    EXPECT_EQ(func(max_normal), 4);
    EXPECT_EQ(func(neg_max_normal), 4);
    EXPECT_EQ(func(min_denormal), 3);
    EXPECT_EQ(func(neg_min_denormal), 3);
    EXPECT_EQ(func(max_denormal), 3);
    EXPECT_EQ(func(zero), 2);
    EXPECT_EQ(func(neg_zero), 2);
  }
};

#define FP_NAN 0
#define FP_INFINITE 1
#define FP_ZERO 2
#define FP_SUBNORMAL 3
#define FP_NORMAL 4

#define LIST_FPCLASSIFY_TESTS(T, func)                                         \
  using LlvmLibcFpClassifyTest = FpClassifyTest<T>;                            \
  TEST_F(LlvmLibcFpClassifyTest, SpecialNumbers) {                             \
    auto fpclassify_func = [](T x) { return func(x); };                        \
    testSpecialNumbers(fpclassify_func);                                       \
  }

#endif // LLVM_LIBC_TEST_INCLUDE_MATH_FPCLASSIFY_H
