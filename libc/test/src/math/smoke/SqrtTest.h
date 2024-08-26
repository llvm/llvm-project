//===-- Utility class to test sqrt[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename OutType, typename InType>
class SqrtTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(OutType)

public:
  typedef OutType (*SqrtFunc)(InType);

  void test_special_numbers(SqrtFunc func) {
    ASSERT_FP_EQ(aNaN, func(aNaN));
    ASSERT_FP_EQ(inf, func(inf));
    ASSERT_FP_EQ(aNaN, func(neg_inf));
    ASSERT_FP_EQ(zero, func(zero));
    ASSERT_FP_EQ(neg_zero, func(neg_zero));
    ASSERT_FP_EQ(aNaN, func(InType(-1.0)));
    ASSERT_FP_EQ(OutType(1.0), func(InType(1.0)));
    ASSERT_FP_EQ(OutType(2.0), func(InType(4.0)));
    ASSERT_FP_EQ(OutType(3.0), func(InType(9.0)));
  }
};

#define LIST_SQRT_TESTS(T, func)                                               \
  using LlvmLibcSqrtTest = SqrtTest<T, T>;                                     \
  TEST_F(LlvmLibcSqrtTest, SpecialNumbers) { test_special_numbers(&func); }

#define LIST_NARROWING_SQRT_TESTS(OutType, InType, func)                       \
  using LlvmLibcSqrtTest = SqrtTest<OutType, InType>;                          \
  TEST_F(LlvmLibcSqrtTest, SpecialNumbers) { test_special_numbers(&func); }
