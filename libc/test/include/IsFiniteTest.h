//===-- Utility class to test the isfinite macro [f|l] ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_INCLUDE_MATH_ISFINITE_H
#define LLVM_LIBC_TEST_INCLUDE_MATH_ISFINITE_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/math-function-macros.h"

#define PI 3.14159265358979323846

template <typename T>
class IsFiniteTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

    DECLARE_SPECIAL_CONSTANTS(T)

  public:
    typedef bool (*IsFiniteFunc)(T);

    void testSpecialNumbers(IsFiniteFunc func) {
      EXPECT_EQ(isfinite(zero), 1); 
      EXPECT_EQ(isfinite(PI), 1); 
      EXPECT_EQ(isfinite(inf), 0); 
      EXPECT_EQ(isfinite(aNaN), 0); 

      EXPECT_EQ(isfinite(neg_zero), 1); 
      EXPECT_EQ(isfinite(-PI), 1); 
      EXPECT_EQ(isfinite(neg_inf), 0); 
      EXPECT_EQ(isfinite(neg_aNaN), 0); 
    }   

    void testSpecialCases(IsFiniteFunc func) {
      EXPECT_EQ(isfinite(PI / zero), 0);       // division by zero
      EXPECT_EQ(isfinite(PI / inf), 1);        // division by +inf
      EXPECT_EQ(isfinite(PI / neg_inf), 1);    // division by -inf
      EXPECT_EQ(isfinite(inf / neg_inf), 0);   // +inf divided by -inf

      EXPECT_EQ(isfinite(inf * neg_inf), 0);   // multiply +inf by -inf
      EXPECT_EQ(isfinite(inf * zero), 0);      // multiply by +inf
      EXPECT_EQ(isfinite(neg_inf * zero), 0);  // multiply by -inf

      EXPECT_EQ(isfinite(inf + neg_inf), 0);   // +inf + -inf
    }   
};

#define LIST_ISFINITE_TESTS(T, func)                                           \
  using LlvmLibcIsFiniteTest = IsFiniteTest<T>;                                \
  TEST_F(LlvmLibcIsFiniteTest, SpecialNumbers) { testSpecialNumbers(&func); }  \
  TEST_F(LlvmLibcIsFiniteTest, SpecialCases) { testSpecialCases(&func); }


#endif // LLVM_LIBC_TEST_INCLUDE_MATH_ISFINITE_H
