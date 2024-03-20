//===-- Utility class to test different flavors of fromfp -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_TEST_SRC_MATH_SMOKE_FROMFPTEST_H
#define LIBC_TEST_SRC_MATH_SMOKE_FROMFPTEST_H

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class FromfpTestTemplate : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*FromfpFunc)(T, int, unsigned int);

  void testSpecialNumbersNonzeroWidth(FromfpFunc func) {
    for (int rnd : MATH_ROUNDING_DIRECTIONS) {
      EXPECT_FP_EQ(zero, func(zero, rnd, 32U));
      EXPECT_FP_EQ(neg_zero, func(neg_zero, rnd, 32U));

      EXPECT_FP_EQ(aNaN, func(inf, rnd, 32U));
      EXPECT_FP_EQ(aNaN, func(neg_inf, rnd, 32U));

      EXPECT_FP_EQ(aNaN, func(aNaN, rnd, 32U));
    }
  }

  void testSpecialNumbersZeroWidth(FromfpFunc func) {
    for (int rnd : MATH_ROUNDING_DIRECTIONS) {
      EXPECT_FP_EQ(aNaN, func(zero, rnd, 0U));
      EXPECT_FP_EQ(aNaN, func(neg_zero, rnd, 0U));

      EXPECT_FP_EQ(aNaN, func(inf, rnd, 0U));
      EXPECT_FP_EQ(aNaN, func(neg_inf, rnd, 0U));

      EXPECT_FP_EQ(aNaN, func(aNaN, rnd, 0U));
    }
  }

  void testRoundedNumbersWithinRange(FromfpFunc func) {
    for (int rnd : MATH_ROUNDING_DIRECTIONS) {
      EXPECT_FP_EQ(T(1.0), func(T(1.0), rnd, 2U));
      EXPECT_FP_EQ(T(-1.0), func(T(-1.0), rnd, 1U));
      EXPECT_FP_EQ(T(10.0), func(T(10.0), rnd, 5U));
      EXPECT_FP_EQ(T(-10.0), func(T(-10.0), rnd, 5U));
      EXPECT_FP_EQ(T(1234.0), func(T(1234.0), rnd, 12U));
      EXPECT_FP_EQ(T(-1234.0), func(T(-1234.0), rnd, 12U));
    }
  }

  void testRoundedNumbersOutsideRange(FromfpFunc func) {
    for (int rnd : MATH_ROUNDING_DIRECTIONS) {
      EXPECT_FP_EQ(aNaN, func(T(1.0), rnd, 1U));
      EXPECT_FP_EQ(aNaN, func(T(10.0), rnd, 4U));
      EXPECT_FP_EQ(aNaN, func(T(-10.0), rnd, 4U));
      EXPECT_FP_EQ(aNaN, func(T(1234.0), rnd, 11U));
      EXPECT_FP_EQ(aNaN, func(T(-1234.0), rnd, 11U));
    }
  }

  void testFractionsUpwardWithinRange(FromfpFunc func) {
    EXPECT_FP_EQ(T(1.0), func(T(0.5), FP_INT_UPWARD, 2U));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.5), FP_INT_UPWARD, 1U));
    EXPECT_FP_EQ(T(1.0), func(T(0.115), FP_INT_UPWARD, 2U));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.115), FP_INT_UPWARD, 1U));
    EXPECT_FP_EQ(T(1.0), func(T(0.715), FP_INT_UPWARD, 2U));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.715), FP_INT_UPWARD, 1U));
    EXPECT_FP_EQ(T(2.0), func(T(1.3), FP_INT_UPWARD, 3U));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.3), FP_INT_UPWARD, 1U));
    EXPECT_FP_EQ(T(2.0), func(T(1.5), FP_INT_UPWARD, 3U));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.5), FP_INT_UPWARD, 1U));
    EXPECT_FP_EQ(T(2.0), func(T(1.75), FP_INT_UPWARD, 3U));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.75), FP_INT_UPWARD, 1U));
    EXPECT_FP_EQ(T(11.0), func(T(10.32), FP_INT_UPWARD, 5U));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.32), FP_INT_UPWARD, 5U));
    EXPECT_FP_EQ(T(11.0), func(T(10.65), FP_INT_UPWARD, 5U));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.65), FP_INT_UPWARD, 5U));
    EXPECT_FP_EQ(T(1235.0), func(T(1234.38), FP_INT_UPWARD, 12U));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.38), FP_INT_UPWARD, 12U));
    EXPECT_FP_EQ(T(1235.0), func(T(1234.96), FP_INT_UPWARD, 12U));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.96), FP_INT_UPWARD, 12U));
  }

  void testFractionsUpwardOutsideRange(FromfpFunc func) {
    EXPECT_FP_EQ(aNaN, func(T(0.5), FP_INT_UPWARD, 1U));
    EXPECT_FP_EQ(aNaN, func(T(0.115), FP_INT_UPWARD, 1U));
    EXPECT_FP_EQ(aNaN, func(T(0.715), FP_INT_UPWARD, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.3), FP_INT_UPWARD, 2U));
    EXPECT_FP_EQ(aNaN, func(T(1.5), FP_INT_UPWARD, 2U));
    EXPECT_FP_EQ(aNaN, func(T(1.75), FP_INT_UPWARD, 2U));
    EXPECT_FP_EQ(aNaN, func(T(10.32), FP_INT_UPWARD, 4U));
    EXPECT_FP_EQ(aNaN, func(T(-10.32), FP_INT_UPWARD, 4U));
    EXPECT_FP_EQ(aNaN, func(T(10.65), FP_INT_UPWARD, 4U));
    EXPECT_FP_EQ(aNaN, func(T(-10.65), FP_INT_UPWARD, 4U));
    EXPECT_FP_EQ(aNaN, func(T(1234.38), FP_INT_UPWARD, 11U));
    EXPECT_FP_EQ(aNaN, func(T(-1234.38), FP_INT_UPWARD, 11U));
    EXPECT_FP_EQ(aNaN, func(T(1234.96), FP_INT_UPWARD, 11U));
    EXPECT_FP_EQ(aNaN, func(T(-1234.96), FP_INT_UPWARD, 11U));
  }

  void testFractionsDownwardWithinRange(FromfpFunc func) {
    EXPECT_FP_EQ(T(0.0), func(T(0.5), FP_INT_DOWNWARD, 1U));
    EXPECT_FP_EQ(T(-1.0), func(T(-0.5), FP_INT_DOWNWARD, 1U));
    EXPECT_FP_EQ(T(0.0), func(T(0.115), FP_INT_DOWNWARD, 1U));
    EXPECT_FP_EQ(T(-1.0), func(T(-0.115), FP_INT_DOWNWARD, 1U));
    EXPECT_FP_EQ(T(0.0), func(T(0.715), FP_INT_DOWNWARD, 1U));
    EXPECT_FP_EQ(T(-1.0), func(T(-0.715), FP_INT_DOWNWARD, 1U));
    EXPECT_FP_EQ(T(1.0), func(T(1.3), FP_INT_DOWNWARD, 2U));
    EXPECT_FP_EQ(T(-2.0), func(T(-1.3), FP_INT_DOWNWARD, 2U));
    EXPECT_FP_EQ(T(1.0), func(T(1.5), FP_INT_DOWNWARD, 2U));
    EXPECT_FP_EQ(T(-2.0), func(T(-1.5), FP_INT_DOWNWARD, 2U));
    EXPECT_FP_EQ(T(1.0), func(T(1.75), FP_INT_DOWNWARD, 2U));
    EXPECT_FP_EQ(T(-2.0), func(T(-1.75), FP_INT_DOWNWARD, 2U));
    EXPECT_FP_EQ(T(10.0), func(T(10.32), FP_INT_DOWNWARD, 5U));
    EXPECT_FP_EQ(T(-11.0), func(T(-10.32), FP_INT_DOWNWARD, 5U));
    EXPECT_FP_EQ(T(10.0), func(T(10.65), FP_INT_DOWNWARD, 5U));
    EXPECT_FP_EQ(T(-11.0), func(T(-10.65), FP_INT_DOWNWARD, 5U));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.38), FP_INT_DOWNWARD, 12U));
    EXPECT_FP_EQ(T(-1235.0), func(T(-1234.38), FP_INT_DOWNWARD, 12U));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.96), FP_INT_DOWNWARD, 12U));
    EXPECT_FP_EQ(T(-1235.0), func(T(-1234.96), FP_INT_DOWNWARD, 12U));
  }

  void testFractionsDownwardOutsideRange(FromfpFunc func) {
    EXPECT_FP_EQ(aNaN, func(T(1.3), FP_INT_DOWNWARD, 1U));
    EXPECT_FP_EQ(aNaN, func(T(-1.3), FP_INT_DOWNWARD, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.5), FP_INT_DOWNWARD, 1U));
    EXPECT_FP_EQ(aNaN, func(T(-1.5), FP_INT_DOWNWARD, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.75), FP_INT_DOWNWARD, 1U));
    EXPECT_FP_EQ(aNaN, func(T(-1.75), FP_INT_DOWNWARD, 1U));
    EXPECT_FP_EQ(aNaN, func(T(10.32), FP_INT_DOWNWARD, 4U));
    EXPECT_FP_EQ(aNaN, func(T(-10.32), FP_INT_DOWNWARD, 4U));
    EXPECT_FP_EQ(aNaN, func(T(10.65), FP_INT_DOWNWARD, 4U));
    EXPECT_FP_EQ(aNaN, func(T(-10.65), FP_INT_DOWNWARD, 4U));
    EXPECT_FP_EQ(aNaN, func(T(1234.38), FP_INT_DOWNWARD, 11U));
    EXPECT_FP_EQ(aNaN, func(T(-1234.38), FP_INT_DOWNWARD, 11U));
    EXPECT_FP_EQ(aNaN, func(T(1234.96), FP_INT_DOWNWARD, 11U));
    EXPECT_FP_EQ(aNaN, func(T(-1234.96), FP_INT_DOWNWARD, 11U));
  }

  void testFractionsTowardZeroWithinRange(FromfpFunc func) {
    EXPECT_FP_EQ(T(0.0), func(T(0.5), FP_INT_TOWARDZERO, 1U));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.5), FP_INT_TOWARDZERO, 1U));
    EXPECT_FP_EQ(T(0.0), func(T(0.115), FP_INT_TOWARDZERO, 1U));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.115), FP_INT_TOWARDZERO, 1U));
    EXPECT_FP_EQ(T(0.0), func(T(0.715), FP_INT_TOWARDZERO, 1U));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.715), FP_INT_TOWARDZERO, 1U));
    EXPECT_FP_EQ(T(1.0), func(T(1.3), FP_INT_TOWARDZERO, 2U));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.3), FP_INT_TOWARDZERO, 1U));
    EXPECT_FP_EQ(T(1.0), func(T(1.5), FP_INT_TOWARDZERO, 2U));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.5), FP_INT_TOWARDZERO, 1U));
    EXPECT_FP_EQ(T(1.0), func(T(1.75), FP_INT_TOWARDZERO, 2U));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.75), FP_INT_TOWARDZERO, 1U));
    EXPECT_FP_EQ(T(10.0), func(T(10.32), FP_INT_TOWARDZERO, 5U));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.32), FP_INT_TOWARDZERO, 5U));
    EXPECT_FP_EQ(T(10.0), func(T(10.65), FP_INT_TOWARDZERO, 5U));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.65), FP_INT_TOWARDZERO, 5U));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.38), FP_INT_TOWARDZERO, 12U));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.38), FP_INT_TOWARDZERO, 12U));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.96), FP_INT_TOWARDZERO, 12U));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.96), FP_INT_TOWARDZERO, 12U));
  }

  void testFractionsTowardZeroOutsideRange(FromfpFunc func) {
    EXPECT_FP_EQ(aNaN, func(T(1.3), FP_INT_TOWARDZERO, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.5), FP_INT_TOWARDZERO, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.75), FP_INT_TOWARDZERO, 1U));
    EXPECT_FP_EQ(aNaN, func(T(10.32), FP_INT_TOWARDZERO, 4U));
    EXPECT_FP_EQ(aNaN, func(T(-10.32), FP_INT_TOWARDZERO, 4U));
    EXPECT_FP_EQ(aNaN, func(T(10.65), FP_INT_TOWARDZERO, 4U));
    EXPECT_FP_EQ(aNaN, func(T(-10.65), FP_INT_TOWARDZERO, 4U));
    EXPECT_FP_EQ(aNaN, func(T(1234.38), FP_INT_TOWARDZERO, 11U));
    EXPECT_FP_EQ(aNaN, func(T(-1234.38), FP_INT_TOWARDZERO, 11U));
    EXPECT_FP_EQ(aNaN, func(T(1234.96), FP_INT_TOWARDZERO, 11U));
    EXPECT_FP_EQ(aNaN, func(T(-1234.96), FP_INT_TOWARDZERO, 11U));
  }

  void testFractionsToNearestFromZeroWithinRange(FromfpFunc func) {
    EXPECT_FP_EQ(T(1.0), func(T(0.5), FP_INT_TONEARESTFROMZERO, 2U));
    EXPECT_FP_EQ(T(-1.0), func(T(-0.5), FP_INT_TONEARESTFROMZERO, 1U));
    EXPECT_FP_EQ(T(0.0), func(T(0.115), FP_INT_TONEARESTFROMZERO, 1U));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.115), FP_INT_TONEARESTFROMZERO, 1U));
    EXPECT_FP_EQ(T(1.0), func(T(0.715), FP_INT_TONEARESTFROMZERO, 2U));
    EXPECT_FP_EQ(T(-1.0), func(T(-0.715), FP_INT_TONEARESTFROMZERO, 1U));
    EXPECT_FP_EQ(T(1.0), func(T(1.3), FP_INT_TONEARESTFROMZERO, 2U));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.3), FP_INT_TONEARESTFROMZERO, 1U));
    EXPECT_FP_EQ(T(2.0), func(T(1.5), FP_INT_TONEARESTFROMZERO, 3U));
    EXPECT_FP_EQ(T(-2.0), func(T(-1.5), FP_INT_TONEARESTFROMZERO, 2U));
    EXPECT_FP_EQ(T(2.0), func(T(1.75), FP_INT_TONEARESTFROMZERO, 3U));
    EXPECT_FP_EQ(T(-2.0), func(T(-1.75), FP_INT_TONEARESTFROMZERO, 2U));
    EXPECT_FP_EQ(T(10.0), func(T(10.32), FP_INT_TONEARESTFROMZERO, 5U));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.32), FP_INT_TONEARESTFROMZERO, 5U));
    EXPECT_FP_EQ(T(11.0), func(T(10.65), FP_INT_TONEARESTFROMZERO, 5U));
    EXPECT_FP_EQ(T(-11.0), func(T(-10.65), FP_INT_TONEARESTFROMZERO, 5U));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.38), FP_INT_TONEARESTFROMZERO, 12U));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.38), FP_INT_TONEARESTFROMZERO, 12U));
    EXPECT_FP_EQ(T(1235.0), func(T(1234.96), FP_INT_TONEARESTFROMZERO, 12U));
    EXPECT_FP_EQ(T(-1235.0), func(T(-1234.96), FP_INT_TONEARESTFROMZERO, 12U));
  }

  void testFractionsToNearestFromZeroOutsideRange(FromfpFunc func) {
    EXPECT_FP_EQ(aNaN, func(T(0.5), FP_INT_TONEARESTFROMZERO, 1U));
    EXPECT_FP_EQ(aNaN, func(T(0.715), FP_INT_TONEARESTFROMZERO, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.3), FP_INT_TONEARESTFROMZERO, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.5), FP_INT_TONEARESTFROMZERO, 2U));
    EXPECT_FP_EQ(aNaN, func(T(-1.5), FP_INT_TONEARESTFROMZERO, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.75), FP_INT_TONEARESTFROMZERO, 2U));
    EXPECT_FP_EQ(aNaN, func(T(-1.75), FP_INT_TONEARESTFROMZERO, 1U));
    EXPECT_FP_EQ(aNaN, func(T(10.32), FP_INT_TONEARESTFROMZERO, 4U));
    EXPECT_FP_EQ(aNaN, func(T(-10.32), FP_INT_TONEARESTFROMZERO, 4U));
    EXPECT_FP_EQ(aNaN, func(T(10.65), FP_INT_TONEARESTFROMZERO, 4U));
    EXPECT_FP_EQ(aNaN, func(T(-10.65), FP_INT_TONEARESTFROMZERO, 4U));
    EXPECT_FP_EQ(aNaN, func(T(1234.38), FP_INT_TONEARESTFROMZERO, 11U));
    EXPECT_FP_EQ(aNaN, func(T(-1234.38), FP_INT_TONEARESTFROMZERO, 11U));
    EXPECT_FP_EQ(aNaN, func(T(1234.96), FP_INT_TONEARESTFROMZERO, 11U));
    EXPECT_FP_EQ(aNaN, func(T(-1234.96), FP_INT_TONEARESTFROMZERO, 11U));
  }

  void testFractionsToNearestWithinRange(FromfpFunc func) {
    EXPECT_FP_EQ(T(0.0), func(T(0.5), FP_INT_TONEAREST, 1U));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.5), FP_INT_TONEAREST, 1U));
    EXPECT_FP_EQ(T(0.0), func(T(0.115), FP_INT_TONEAREST, 1U));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.115), FP_INT_TONEAREST, 1U));
    EXPECT_FP_EQ(T(1.0), func(T(0.715), FP_INT_TONEAREST, 2U));
    EXPECT_FP_EQ(T(-1.0), func(T(-0.715), FP_INT_TONEAREST, 1U));
    EXPECT_FP_EQ(T(1.0), func(T(1.3), FP_INT_TONEAREST, 2U));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.3), FP_INT_TONEAREST, 1U));
    EXPECT_FP_EQ(T(2.0), func(T(1.5), FP_INT_TONEAREST, 3U));
    EXPECT_FP_EQ(T(-2.0), func(T(-1.5), FP_INT_TONEAREST, 2U));
    EXPECT_FP_EQ(T(2.0), func(T(1.75), FP_INT_TONEAREST, 3U));
    EXPECT_FP_EQ(T(-2.0), func(T(-1.75), FP_INT_TONEAREST, 2U));
    EXPECT_FP_EQ(T(10.0), func(T(10.32), FP_INT_TONEAREST, 5U));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.32), FP_INT_TONEAREST, 5U));
    EXPECT_FP_EQ(T(11.0), func(T(10.65), FP_INT_TONEAREST, 5U));
    EXPECT_FP_EQ(T(-11.0), func(T(-10.65), FP_INT_TONEAREST, 5U));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.38), FP_INT_TONEAREST, 12U));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.38), FP_INT_TONEAREST, 12U));
    EXPECT_FP_EQ(T(1235.0), func(T(1234.96), FP_INT_TONEAREST, 12U));
    EXPECT_FP_EQ(T(-1235.0), func(T(-1234.96), FP_INT_TONEAREST, 12U));

    EXPECT_FP_EQ(T(2.0), func(T(2.3), FP_INT_TONEAREST, 3U));
    EXPECT_FP_EQ(T(-2.0), func(T(-2.3), FP_INT_TONEAREST, 2U));
    EXPECT_FP_EQ(T(2.0), func(T(2.5), FP_INT_TONEAREST, 3U));
    EXPECT_FP_EQ(T(-2.0), func(T(-2.5), FP_INT_TONEAREST, 2U));
    EXPECT_FP_EQ(T(3.0), func(T(2.75), FP_INT_TONEAREST, 3U));
    EXPECT_FP_EQ(T(-3.0), func(T(-2.75), FP_INT_TONEAREST, 3U));
    EXPECT_FP_EQ(T(5.0), func(T(5.3), FP_INT_TONEAREST, 4U));
    EXPECT_FP_EQ(T(-5.0), func(T(-5.3), FP_INT_TONEAREST, 4U));
    EXPECT_FP_EQ(T(6.0), func(T(5.5), FP_INT_TONEAREST, 4U));
    EXPECT_FP_EQ(T(-6.0), func(T(-5.5), FP_INT_TONEAREST, 4U));
    EXPECT_FP_EQ(T(6.0), func(T(5.75), FP_INT_TONEAREST, 4U));
    EXPECT_FP_EQ(T(-6.0), func(T(-5.75), FP_INT_TONEAREST, 4U));
  }

  void testFractionsToNearestOutsideRange(FromfpFunc func) {
    EXPECT_FP_EQ(aNaN, func(T(0.715), FP_INT_TONEAREST, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.3), FP_INT_TONEAREST, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.5), FP_INT_TONEAREST, 2U));
    EXPECT_FP_EQ(aNaN, func(T(-1.5), FP_INT_TONEAREST, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.75), FP_INT_TONEAREST, 2U));
    EXPECT_FP_EQ(aNaN, func(T(-1.75), FP_INT_TONEAREST, 1U));
    EXPECT_FP_EQ(aNaN, func(T(10.32), FP_INT_TONEAREST, 4U));
    EXPECT_FP_EQ(aNaN, func(T(-10.32), FP_INT_TONEAREST, 4U));
    EXPECT_FP_EQ(aNaN, func(T(10.65), FP_INT_TONEAREST, 4U));
    EXPECT_FP_EQ(aNaN, func(T(-10.65), FP_INT_TONEAREST, 4U));
    EXPECT_FP_EQ(aNaN, func(T(1234.38), FP_INT_TONEAREST, 11U));
    EXPECT_FP_EQ(aNaN, func(T(-1234.38), FP_INT_TONEAREST, 11U));
    EXPECT_FP_EQ(aNaN, func(T(1234.96), FP_INT_TONEAREST, 11U));
    EXPECT_FP_EQ(aNaN, func(T(-1234.96), FP_INT_TONEAREST, 11U));

    EXPECT_FP_EQ(aNaN, func(T(2.3), FP_INT_TONEAREST, 2U));
    EXPECT_FP_EQ(aNaN, func(T(-2.3), FP_INT_TONEAREST, 1U));
    EXPECT_FP_EQ(aNaN, func(T(2.5), FP_INT_TONEAREST, 2U));
    EXPECT_FP_EQ(aNaN, func(T(-2.5), FP_INT_TONEAREST, 1U));
    EXPECT_FP_EQ(aNaN, func(T(2.75), FP_INT_TONEAREST, 2U));
    EXPECT_FP_EQ(aNaN, func(T(-2.75), FP_INT_TONEAREST, 2U));
    EXPECT_FP_EQ(aNaN, func(T(5.3), FP_INT_TONEAREST, 3U));
    EXPECT_FP_EQ(aNaN, func(T(-5.3), FP_INT_TONEAREST, 3U));
    EXPECT_FP_EQ(aNaN, func(T(5.5), FP_INT_TONEAREST, 3U));
    EXPECT_FP_EQ(aNaN, func(T(-5.5), FP_INT_TONEAREST, 3U));
    EXPECT_FP_EQ(aNaN, func(T(5.75), FP_INT_TONEAREST, 3U));
    EXPECT_FP_EQ(aNaN, func(T(-5.75), FP_INT_TONEAREST, 3U));
  }
};

#define LIST_FROMFP_TESTS(T, func)                                             \
  using LlvmLibcFromfpTest = FromfpTestTemplate<T>;                            \
  TEST_F(LlvmLibcFromfpTest, SpecialNumbersNonzeroWidth) {                     \
    testSpecialNumbersNonzeroWidth(&func);                                     \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, SpecialNumbersZeroWidth) {                        \
    testSpecialNumbersZeroWidth(&func);                                        \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, RoundedNumbersWithinRange) {                      \
    testRoundedNumbersWithinRange(&func);                                      \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, RoundedNumbersOutsideRange) {                     \
    testRoundedNumbersOutsideRange(&func);                                     \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, FractionsUpwardWithinRange) {                     \
    testFractionsUpwardWithinRange(&func);                                     \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, FractionsUpwardOutsideRange) {                    \
    testFractionsUpwardOutsideRange(&func);                                    \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, FractionsDownwardWithinRange) {                   \
    testFractionsDownwardWithinRange(&func);                                   \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, FractionsDownwardOutsideRange) {                  \
    testFractionsDownwardOutsideRange(&func);                                  \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, FractionsTowardZeroWithinRange) {                 \
    testFractionsTowardZeroWithinRange(&func);                                 \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, FractionsTowardZeroOutsideRange) {                \
    testFractionsTowardZeroOutsideRange(&func);                                \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, FractionsToNearestFromZeroWithinRange) {          \
    testFractionsToNearestFromZeroWithinRange(&func);                          \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, FractionsToNearestFromZeroOutsideRange) {         \
    testFractionsToNearestFromZeroOutsideRange(&func);                         \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, FractionsToNearestWithinRange) {                  \
    testFractionsToNearestWithinRange(&func);                                  \
  }                                                                            \
  TEST_F(LlvmLibcFromfpTest, FractionsToNearestOutsideRange) {                 \
    testFractionsToNearestOutsideRange(&func);                                 \
  }

#endif // LIBC_TEST_SRC_MATH_SMOKE_FROMFPTEST_H
