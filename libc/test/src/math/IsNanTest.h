//===-- Utility class to test isnan[f|l] ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class IsNanTest : public LIBC_NAMESPACE::testing::FPTest<T> {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef int (*IsNanFunc)(T);

  void testSpecialNumbers(IsNanFunc func) {
    EXPECT_EQ(func(zero), 0);
    EXPECT_EQ(func(neg_zero), 0);
    EXPECT_EQ(func(inf), 0);
    EXPECT_EQ(func(neg_inf), 0);
    EXPECT_EQ(func(aNaN), 1);
    EXPECT_EQ(func(sNaN), 1);
  }
};

#define LIST_ISNAN_TESTS(T, func)                                              \
  using LlvmLibcIsNanTest = IsNanTest<T>;                                      \
  TEST_F(LlvmLibcIsNanTest, SpecialNumbers) { testSpecialNumbers(&func); }
