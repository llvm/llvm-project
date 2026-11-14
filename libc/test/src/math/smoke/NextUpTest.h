//===-- Utility class to test different flavors of nextup -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_NEXTUPTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_NEXTUPTEST_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class NextUpTestTemplate : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*NextUpFunc)(T);

  void testNaN(NextUpFunc func) { ASSERT_FP_EQ(func(aNaN), aNaN); }

  void testBoundaries(NextUpFunc func) {
    EXPECT_FP_EQ_WITH_EXCEPTION(neg_zero, func(neg_min_denormal), 0);

    ASSERT_FP_EQ(min_denormal, func(zero));
    ASSERT_FP_EQ(min_denormal, func(neg_zero));

    EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(max_normal), 0);
    ASSERT_FP_EQ(inf, func(inf));

    ASSERT_FP_EQ(neg_max_normal, func(neg_inf));

    EXPECT_FP_EQ_WITH_EXCEPTION(FPBits::min_normal().get_val(),
                                func(FPBits::max_subnormal().get_val()), 0);
    EXPECT_FP_EQ_WITH_EXCEPTION(FPBits::max_subnormal(Sign::NEG).get_val(),
                                func(FPBits::min_normal(Sign::NEG).get_val()),
                                0);
  }
};

#define LIST_NEXTUP_TESTS(Name, T, func)                                       \
  using LlvmLibc##Name##Test = NextUpTestTemplate<T>;                          \
  TEST_F(LlvmLibc##Name##Test, TestNaN) { testNaN(&func); }                    \
  TEST_F(LlvmLibc##Name##Test, TestBoundaries) { testBoundaries(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_NEXTUPTEST_H
