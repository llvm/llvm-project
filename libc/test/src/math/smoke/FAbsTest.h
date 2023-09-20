//===-- Utility class to test fabs[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <math.h>

template <typename T> class FAbsTest : public __llvm_libc::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*FabsFunc)(T);

  void testSpecialNumbers(FabsFunc func) {
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, func(aNaN));

    EXPECT_FP_EQ_ALL_ROUNDING(inf, func(inf));
    EXPECT_FP_EQ_ALL_ROUNDING(inf, func(neg_inf));

    EXPECT_FP_EQ_ALL_ROUNDING(zero, func(zero));
    EXPECT_FP_EQ_ALL_ROUNDING(zero, func(neg_zero));

    EXPECT_FP_EQ_ALL_ROUNDING(T(1), func(T(1)));
    EXPECT_FP_EQ_ALL_ROUNDING(T(1), func(T(-1)));
  }
};

#define LIST_FABS_TESTS(T, func)                                               \
  using LlvmLibcFAbsTest = FAbsTest<T>;                                        \
  TEST_F(LlvmLibcFAbsTest, SpecialNumbers) { testSpecialNumbers(&func); }
