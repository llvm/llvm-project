//===-- Utility class to test the issignaling macro  ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_INCLUDE_MATH_ISSIGNALING_H
#define LLVM_LIBC_TEST_INCLUDE_MATH_ISSIGNALING_H

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/math-function-macros.h"

template <typename T>
class IsSignalingTest : public LIBC_NAMESPACE::testing::Test {
  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef int (*IsSignalingFunc)(T);

  void testSpecialNumbers(IsSignalingFunc func) {
    EXPECT_EQ(func(aNaN), 0);
    EXPECT_EQ(func(neg_aNaN), 0);
    EXPECT_EQ(func(sNaN), 1);
    EXPECT_EQ(func(neg_sNaN), 1);
    EXPECT_EQ(func(inf), 0);
    EXPECT_EQ(func(neg_inf), 0);
    EXPECT_EQ(func(min_normal), 0);
    EXPECT_EQ(func(max_normal), 0);
    EXPECT_EQ(func(neg_max_normal), 0);
    EXPECT_EQ(func(min_denormal), 0);
    EXPECT_EQ(func(neg_min_denormal), 0);
    EXPECT_EQ(func(max_denormal), 0);
    EXPECT_EQ(func(zero), 0);
    EXPECT_EQ(func(neg_zero), 0);
  }
};

#define LIST_ISSIGNALING_TESTS(T, func)                                        \
  using LlvmLibcIsSignalingTest = IsSignalingTest<T>;                          \
  TEST_F(LlvmLibcIsSignalingTest, SpecialNumbers) {                            \
    auto issignaling_func = [](T x) { return func(x); };                       \
    testSpecialNumbers(issignaling_func);                                      \
  }

#endif // LLVM_LIBC_TEST_INCLUDE_MATH_ISSIGNALING_H
