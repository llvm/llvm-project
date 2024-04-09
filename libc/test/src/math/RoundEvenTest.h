//===-- Utility class to test roundeven[f|l] --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_ROUNDEVENTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_ROUNDEVENTEST_H

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include "hdr/math_macros.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename T>
class RoundEvenTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*RoundEvenFunc)(T);

  void testSpecialNumbers(RoundEvenFunc func) {
    EXPECT_FP_EQ(zero, func(zero));
    EXPECT_FP_EQ(neg_zero, func(neg_zero));

    EXPECT_FP_EQ(inf, func(inf));
    EXPECT_FP_EQ(neg_inf, func(neg_inf));

    EXPECT_FP_EQ(aNaN, func(aNaN));
  }

  void testRoundedNumbers(RoundEvenFunc func) {
    EXPECT_FP_EQ(T(1.0), func(T(1.0)));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.0)));
    EXPECT_FP_EQ(T(10.0), func(T(10.0)));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.0)));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.0)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.0)));
  }

  void testFractions(RoundEvenFunc func) {
    EXPECT_FP_EQ(T(0.0), func(T(0.5)));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.5)));
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
    EXPECT_FP_EQ(T(11.0), func(T(10.65)));
    EXPECT_FP_EQ(T(-11.0), func(T(-10.65)));
    EXPECT_FP_EQ(T(1233.0), func(T(1233.25)));
    EXPECT_FP_EQ(T(1234.0), func(T(1233.50)));
    EXPECT_FP_EQ(T(1234.0), func(T(1233.75)));
    EXPECT_FP_EQ(T(-1233.0), func(T(-1233.25)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1233.50)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1233.75)));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.50)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.50)));
  }

  void testRange(RoundEvenFunc func) {
    constexpr StorageType COUNT = 100'000;
    constexpr StorageType STEP = STORAGE_MAX / COUNT;
    for (StorageType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      T x = FPBits(v).get_val();
      if (isnan(x) || isinf(x))
        continue;

      ASSERT_MPFR_MATCH(mpfr::Operation::RoundEven, x, func(x), 0.0);
    }
  }
};

#define LIST_ROUNDEVEN_TESTS(T, func)                                          \
  using LlvmLibcRoundEvenTest = RoundEvenTest<T>;                              \
  TEST_F(LlvmLibcRoundEvenTest, SpecialNumbers) { testSpecialNumbers(&func); } \
  TEST_F(LlvmLibcRoundEvenTest, RoundedNubmers) { testRoundedNumbers(&func); } \
  TEST_F(LlvmLibcRoundEvenTest, Fractions) { testFractions(&func); }           \
  TEST_F(LlvmLibcRoundEvenTest, Range) { testRange(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_ROUNDEVENTEST_H
