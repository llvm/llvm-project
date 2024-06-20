//===-- Utility class to test different flavors of float div --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_DIVTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_DIVTEST_H

#include "hdr/fenv_macros.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename OutType, typename InType>
class DivTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(OutType)

public:
  typedef OutType (*DivFunc)(InType, InType);

  void test_special_numbers(DivFunc func) {
    EXPECT_FP_IS_NAN(func(aNaN, aNaN));

    EXPECT_FP_EQ(inf, func(inf, zero));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, zero));
    EXPECT_FP_EQ(neg_inf, func(inf, neg_zero));
    EXPECT_FP_EQ(inf, func(neg_inf, neg_zero));

    EXPECT_FP_EQ(inf, func(inf, zero));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, zero));
    EXPECT_FP_EQ(neg_inf, func(inf, neg_zero));
    EXPECT_FP_EQ(inf, func(neg_inf, neg_zero));
  }

  void test_division_by_zero(DivFunc func) {
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(InType(1.0), zero), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(InType(-1.0), zero),
                                FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(InType(1.0), neg_zero),
                                FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(InType(1.0), zero), FE_DIVBYZERO);
  }

  void test_invalid_operations(DivFunc func) {
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(zero, zero), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_zero, zero), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(zero, neg_zero), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_zero, neg_zero), FE_INVALID);

    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(inf, inf), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_inf, inf), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(inf, neg_inf), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_inf, neg_inf), FE_INVALID);
  }
};

#define LIST_DIV_TESTS(OutType, InType, func)                                  \
  using LlvmLibcDivTest = DivTest<OutType, InType>;                            \
  TEST_F(LlvmLibcDivTest, SpecialNumbers) { test_special_numbers(&func); }     \
  TEST_F(LlvmLibcDivTest, DivisionByZero) { test_division_by_zero(&func); }    \
  TEST_F(LlvmLibcDivTest, InvalidOperations) { test_invalid_operations(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_DIVTEST_H
