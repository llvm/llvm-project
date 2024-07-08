//===-- Utility class to test the isinf macro [f|l] -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_INCLUDE_MATH_ISINF_H
#define LLVM_LIBC_TEST_INCLUDE_MATH_ISINF_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/math-function-macros.h"

#define PI 3.14159265358979323846

template <typename T>
class IsInfTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

    DECLARE_SPECIAL_CONSTANTS(T)

  public:
    typedef bool (*IsInfFunc)(T);

    void testSpecialNumbers(IsInfFunc func) {
      EXPECT_EQ(isinf(zero), 0); 
      EXPECT_EQ(isinf(PI), 0); 
      EXPECT_EQ(isinf(inf), 1); 
      EXPECT_EQ(isinf(aNaN), 0); 

      EXPECT_EQ(isinf(neg_zero), 0); 
      EXPECT_EQ(isinf(-PI), 0); 
      EXPECT_EQ(isinf(neg_inf), 1); 
      EXPECT_EQ(isinf(neg_aNaN), 0); 
    }   

    void testSpecialCases(IsInfFunc func) {
      EXPECT_EQ(isinf(PI / zero), 1);       // division by zero
      EXPECT_EQ(isinf(PI / inf), 0);        // division by +inf
      EXPECT_EQ(isinf(PI / neg_inf), 0);    // division by -inf
      EXPECT_EQ(isinf(inf / neg_inf), 0);   // +inf divided by -inf

      EXPECT_EQ(isinf(inf * neg_inf), 1);   // multiply +inf by -inf
      EXPECT_EQ(isinf(inf * zero), 0);      // multiply by +inf
      EXPECT_EQ(isinf(neg_inf * zero), 0);  // multiply by -inf

      EXPECT_EQ(isinf(inf + neg_inf), 0);   // +inf + -inf
    }   
};

#define LIST_ISINF_TESTS(T, func)                                              \
  using LlvmLibcIsInfTest = IsInfTest<T>;                                      \
  TEST_F(LlvmLibcIsInfTest, SpecialNumbers) { testSpecialNumbers(&func); }     \
  TEST_F(LlvmLibcIsInfTest, SpecialCases) { testSpecialCases(&func); }


#endif // LLVM_LIBC_TEST_INCLUDE_MATH_ISINF_H
