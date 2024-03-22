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

#include "include/llvm-libc-macros/math-macros.h"

#define TEST_SPECIAL(x, y, expected, expected_exception)                       \
  EXPECT_FP_EQ(expected, f(&x, &y));                                           \
  EXPECT_FP_EXCEPTION(expected_exception);                                     \
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT)

#define TEST_REGULAR(x, y, expected) TEST_SPECIAL(x, y, expected, 0)

template <typename T>
class CanonicalizeTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*CanonicalizeFunc)(T *, T *);

  void testSpecialNumbers(CanonicalizeFunc f) {
    T cx;
    TEST_SPECIAL(cx, zero, 0, 0);
    EXPECT_EQ(cx, T(0.0));
    TEST_SPECIAL(cx, neg_zero, 0, 0);
    EXPECT_EQ(cx, T(-0.0));
    TEST_SPECIAL(cx, inf, 0, 0);
    EXPECT_EQ(cx, inf);
    TEST_SPECIAL(cx, neg_inf, 0, 0);
    EXPECT_EQ(cx, neg_inf);
    TEST_SPECIAL(cx, sNaN, 1, FE_INVALID);
    EXPECT_EQ(cx, aNaN);
    TEST_SPECIAL(cx, -sNaN, 1, FE_INVALID);
    EXPECT_EQ(cx, -aNaN);
  }

  void testRegularNumbers(CanonicalizeFunc func) {
    T cx;
    TEST_REGULAR(cx, T(1.0), 0);
    EXPECT_EQ(cx, T(1.0));
    TEST_REGULAR(cx, T(-1.0), 0);
    EXPECT_EQ(cx, T(-1.0));
    TEST_REGULAR(cx, T(10.0), 0);
    EXPECT_EQ(cx, T(10.0));
    TEST_REGULAR(cx, T(-10.0), 0);
    EXPECT_EQ(cx, T(-10.0));
    TEST_REGULAR(cx, T(1234.0), 0);
    EXPECT_EQ(cx, T(1234.0));
    TEST_REGULAR(cx, T(-1234.0), 0);
    EXPECT_EQ(cx, T(-1234.0));
  }
};

#define LIST_CANONICALIZE_TESTS(T, func)                                       \
  using LlvmLibcCanonicalizeTest = CanonicalizeTest<T>;                        \
  TEST_F(LlvmLibcCanonicalizeTest, SpecialNumbers) {                           \
    testSpecialNumbers(&func);                                                 \
  }                                                                            \
  TEST_F(LlvmLibcCanonicalizeTest, RegularNubmers) {                           \
    testRegularNumbers(&func);                                                 \
  }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_CANONICALIZETEST_H
