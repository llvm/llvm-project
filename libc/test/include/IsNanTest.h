//===-- Utility class to test the isnan macro [f|l] -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_INCLUDE_MATH_ISNAN_H
#define LLVM_LIBC_TEST_INCLUDE_MATH_ISNAN_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/math-function-macros.h"

#define PI 3.14159265358979323846

template <typename T>
class IsNanTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef bool (*IsNanFunc)(T);

  void testSpecialNumbers(IsNanFunc func) {
    EXPECT_EQ(isnan(zero), 0);
    EXPECT_EQ(isnan(PI), 0);
    EXPECT_EQ(isnan(inf), 0);
    EXPECT_EQ(isnan(aNaN), 1);

    EXPECT_EQ(isnan(neg_zero), 0);
    EXPECT_EQ(isnan(-PI), 0);
    EXPECT_EQ(isnan(neg_inf), 0);
    EXPECT_EQ(isnan(neg_aNaN), 1);
  }

  void testSpecialCases(IsNanFunc func) {
    EXPECT_EQ(isnan(PI / zero), 0);     // division by zero
    EXPECT_EQ(isnan(PI / inf), 0);      // division by +inf
    EXPECT_EQ(isnan(PI / neg_inf), 0);  // division by -inf
    EXPECT_EQ(isnan(inf / neg_inf), 1); // +inf divided by -inf

    EXPECT_EQ(isnan(inf * neg_inf), 0);  // multiply +inf by -inf
    EXPECT_EQ(isnan(inf * zero), 1);     // multiply by +inf
    EXPECT_EQ(isnan(neg_inf * zero), 1); // multiply by -inf

    EXPECT_EQ(isnan(inf + neg_inf), 1); // +inf + -inf
  }
};

#define LIST_ISNAN_TESTS(T, func)                                              \
  using LlvmLibcIsNanTest = IsNanTest<T>;                                      \
  TEST_F(LlvmLibcIsNanTest, SpecialNumbers) { testSpecialNumbers(&func); }     \
  TEST_F(LlvmLibcIsNanTest, SpecialCases) { testSpecialCases(&func); }

#endif // LLVM_LIBC_TEST_INCLUDE_MATH_ISNAN_H
