//===-- Utility class to test rsqrt[f|l] ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_RSQRTTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_RSQRTTEST_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class RsqrtTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*RsqrtFunc)(T);

  void test_special_numbers(RsqrtFunc func) {
    ASSERT_FP_EQ(aNaN, func(aNaN));
    ASSERT_FP_EQ(inf, func(zero));
    ASSERT_FP_EQ(neg_inf, func(neg_zero));
    ASSERT_FP_EQ(zero, func(inf));
    ASSERT_FP_EQ(aNaN, func(neg_inf));
    ASSERT_FP_EQ(aNaN, func(T(-1.0)));
    ASSERT_FP_EQ(T(1.0), func(T(1.0)));
    ASSERT_FP_EQ(T(0.5), func(T(4.0)));
    ASSERT_FP_EQ(T(0.70710677), func(T(2.0)));
  }
};

#define LIST_RSQRT_TESTS(Name, T, func)                                        \
  using LlvmLibc##Name##Test = RsqrtTest<T>;                                   \
  TEST_F(LlvmLibc##Name##Test, SpecialNumbers) {                               \
    test_special_numbers(&func);                                               \
  }                                                                            \
  static_assert(true, "Require semicolon.")

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_RSQRTTEST_H
