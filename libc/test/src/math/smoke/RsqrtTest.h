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

template <typename OutType, typename InType>
class RsqrtTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(OutType)

  struct InConstants {
    DECLARE_SPECIAL_CONSTANTS(InType)
  };

  InConstants in;

public:
  typedef OutType (*RsqrtFunc)(InType);

  void test_special_numbers(RsqrtFunc func) {
    ASSERT_FP_EQ(in.aNaN, func(in.aNaN));
    ASSERT_FP_EQ(inf, func(in.zero));
    ASSERT_FP_EQ(neg_inf, func(in.neg_zero));
    ASSERT_FP_EQ(zero, func(in.inf));
    ASSERT_FP_EQ(aNaN, func(in.neg_inf));
    ASSERT_FP_EQ(aNaN, func(InType(-1.0)));
    ASSERT_FP_EQ(OutType(1.0), func(InType(1.0)));
    ASSERT_FP_EQ(OutType(0.5), func(InType(4.0)));
    ASSERT_FP_EQ(OutType(0.70710677), func(InType(2.0)));
  }
};

#define LIST_RSQRT_TESTS(Name, T, func)                                        \
  using LlvmLibc##Name##Test = RsqrtTest<T, T>;                                \
  TEST_F(LlvmLibc##Name##Test, SpecialNumbers) {                               \
    test_special_numbers(&func);                                               \
  }                                                                            \
  static_assert(true, "Require semicolon.")

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_RSQRTTEST_H