//===-- Utility class to test sqrt[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/bit.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <math.h>

template <typename T> class SqrtTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

  static constexpr UIntType HIDDEN_BIT =
      UIntType(1) << LIBC_NAMESPACE::fputil::MantissaWidth<T>::VALUE;

public:
  typedef T (*SqrtFunc)(T);

  void test_special_numbers(SqrtFunc func) {
    ASSERT_FP_EQ(aNaN, func(aNaN));
    ASSERT_FP_EQ(inf, func(inf));
    ASSERT_FP_EQ(aNaN, func(neg_inf));
    ASSERT_FP_EQ(0.0, func(0.0));
    ASSERT_FP_EQ(-0.0, func(-0.0));
    ASSERT_FP_EQ(aNaN, func(T(-1.0)));
    ASSERT_FP_EQ(T(1.0), func(T(1.0)));
    ASSERT_FP_EQ(T(2.0), func(T(4.0)));
    ASSERT_FP_EQ(T(3.0), func(T(9.0)));
  }
};

#define LIST_SQRT_TESTS(T, func)                                               \
  using LlvmLibcSqrtTest = SqrtTest<T>;                                        \
  TEST_F(LlvmLibcSqrtTest, SpecialNumbers) { test_special_numbers(&func); }
