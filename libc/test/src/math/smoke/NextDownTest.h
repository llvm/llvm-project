//===-- Utility class to test different flavors of nextdown -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_NEXTDOWNTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_NEXTDOWNTEST_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class NextDownTestTemplate : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*NextDownFunc)(T);

  void testNaN(NextDownFunc func) { ASSERT_FP_EQ(func(aNaN), aNaN); }

  void testBoundaries(NextDownFunc func) {
    EXPECT_FP_EQ_WITH_EXCEPTION(zero, func(min_denormal), 0);

    ASSERT_FP_EQ(neg_min_denormal, func(zero));
    ASSERT_FP_EQ(neg_min_denormal, func(neg_zero));

    EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(neg_max_normal), 0);
    ASSERT_FP_EQ(neg_inf, func(neg_inf));

    ASSERT_FP_EQ(max_normal, func(inf));

    EXPECT_FP_EQ_WITH_EXCEPTION(
        FPBits::min_normal(Sign::NEG).get_val(),
        func(FPBits::max_subnormal(Sign::NEG).get_val()), 0);
    EXPECT_FP_EQ_WITH_EXCEPTION(FPBits::max_subnormal().get_val(),
                                func(FPBits::min_normal().get_val()), 0);
  }
};

#define LIST_NEXTDOWN_TESTS(Name, T, func)                                     \
  using LlvmLibc##Name##Test = NextDownTestTemplate<T>;                        \
  TEST_F(LlvmLibc##Name##Test, TestNaN) { testNaN(&func); }                    \
  TEST_F(LlvmLibc##Name##Test, TestBoundaries) { testBoundaries(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_NEXTDOWNTEST_H
