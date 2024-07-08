//===-- Utility class to test the signbit macro [f|l] -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_INCLUDE_MATH_SIGNBIT_H
#define LLVM_LIBC_TEST_INCLUDE_MATH_SIGNBIT_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/math-function-macros.h"

#define PI 3.14159265358979323846

template <typename T>
class SignbitTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

    DECLARE_SPECIAL_CONSTANTS(T)

  public:
    typedef bool (*SignbitFunc)(T);

    void testSpecialNumbers(SignbitFunc func) {
      EXPECT_EQ(signbit(zero), 0);
      EXPECT_EQ(signbit(PI), 0);
      EXPECT_EQ(signbit(inf), 0); 
      EXPECT_EQ(signbit(aNaN), 0);

      EXPECT_EQ(signbit(neg_zero), 1);
      EXPECT_EQ(signbit(-PI), 1);
      EXPECT_EQ(signbit(neg_inf), 1); 
      EXPECT_EQ(signbit(neg_aNaN), 1);
    }

    void testSpecialCases(SignbitFunc func) {
      EXPECT_EQ(signbit(PI / zero), 0);       // division by zero
      EXPECT_EQ(signbit(PI / inf), 0);        // division by +inf
      EXPECT_EQ(signbit(PI / neg_inf), 1);    // division by -inf
      EXPECT_EQ(signbit(inf / neg_inf), 1);   // +inf divided by -inf

      EXPECT_EQ(signbit(inf * neg_inf), 1);   // multiply +inf by -inf
      EXPECT_EQ(signbit(inf * zero), 1);      // multiply by +inf
      EXPECT_EQ(signbit(neg_inf * zero), 1);  // multiply by -inf

      EXPECT_EQ(signbit(inf + neg_inf), 1);   // +inf + -inf
    }   
};

#define LIST_SIGNBIT_TESTS(T, func)                                            \
  using LlvmLibcSignbitTest = SignbitTest<T>;                                  \
  TEST_F(LlvmLibcSignbitTest, SpecialNumbers) { testSpecialNumbers(&func); }   \
  TEST_F(LlvmLibcSignbitTest, SpecialCases) { testSpecialCases(&func); }


#endif // LLVM_LIBC_TEST_INCLUDE_MATH_SIGNBIT_H
