//===-- Utility class to test round[f|l] ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_ROUNDTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_ROUNDTEST_H

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <math.h>

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename T> class RoundTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*RoundFunc)(T);

  void testSpecialNumbers(RoundFunc func) {
    EXPECT_FP_EQ(zero, func(zero));
    EXPECT_FP_EQ(neg_zero, func(neg_zero));

    EXPECT_FP_EQ(inf, func(inf));
    EXPECT_FP_EQ(neg_inf, func(neg_inf));

    EXPECT_FP_EQ(aNaN, func(aNaN));
  }

  void testRoundedNumbers(RoundFunc func) {
    EXPECT_FP_EQ(T(1.0), func(T(1.0)));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.0)));
    EXPECT_FP_EQ(T(10.0), func(T(10.0)));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.0)));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.0)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.0)));
  }

  void testFractions(RoundFunc func) {
    EXPECT_FP_EQ(T(1.0), func(T(0.5)));
    EXPECT_FP_EQ(T(-1.0), func(T(-0.5)));
    EXPECT_FP_EQ(T(0.0), func(T(0.115)));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.115)));
    EXPECT_FP_EQ(T(1.0), func(T(0.715)));
    EXPECT_FP_EQ(T(-1.0), func(T(-0.715)));
    EXPECT_FP_EQ(T(1.0), func(T(1.3)));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.3)));
    EXPECT_FP_EQ(T(2.0), func(T(1.5)));
    EXPECT_FP_EQ(T(-2.0), func(T(-1.5)));
    EXPECT_FP_EQ(T(2.0), func(T(1.75)));
    EXPECT_FP_EQ(T(-2.0), func(T(-1.75)));
    EXPECT_FP_EQ(T(10.0), func(T(10.32)));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.32)));
    EXPECT_FP_EQ(T(11.0), func(T(10.65)));
    EXPECT_FP_EQ(T(-11.0), func(T(-10.65)));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.38)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.38)));
    EXPECT_FP_EQ(T(1235.0), func(T(1234.96)));
    EXPECT_FP_EQ(T(-1235.0), func(T(-1234.96)));
  }

  void testRange(RoundFunc func) {
    constexpr StorageType COUNT = 100'000;
    constexpr StorageType STEP = STORAGE_MAX / COUNT;
    for (StorageType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      T x = FPBits(v).get_val();
      if (isnan(x) || isinf(x))
        continue;

      ASSERT_MPFR_MATCH(mpfr::Operation::Round, x, func(x), 0.0);
    }
  }
};

#define LIST_ROUND_TESTS(T, func)                                              \
  using LlvmLibcRoundTest = RoundTest<T>;                                      \
  TEST_F(LlvmLibcRoundTest, SpecialNumbers) { testSpecialNumbers(&func); }     \
  TEST_F(LlvmLibcRoundTest, RoundedNubmers) { testRoundedNumbers(&func); }     \
  TEST_F(LlvmLibcRoundTest, Fractions) { testFractions(&func); }               \
  TEST_F(LlvmLibcRoundTest, Range) { testRange(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_ROUNDTEST_H
