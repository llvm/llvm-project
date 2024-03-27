//===-- Utility class to test different flavors of fromfpx ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_TEST_SRC_MATH_SMOKE_FROMFPXTEST_H
#define LIBC_TEST_SRC_MATH_SMOKE_FROMFPXTEST_H

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class FromfpxTestTemplate : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*FromfpxFunc)(T, int, unsigned int);

  void testSpecialNumbersNonzeroWidth(FromfpxFunc func) {
    for (int rnd : MATH_ROUNDING_DIRECTIONS_INCLUDING_UNKNOWN) {
      EXPECT_FP_EQ(zero, func(zero, rnd, 32U));
      EXPECT_FP_EQ(neg_zero, func(neg_zero, rnd, 32U));

      EXPECT_FP_EQ(aNaN, func(inf, rnd, 32U));
      EXPECT_FP_EQ(aNaN, func(neg_inf, rnd, 32U));

      EXPECT_FP_EQ(aNaN, func(aNaN, rnd, 32U));
    }
  }

  void testSpecialNumbersZeroWidth(FromfpxFunc func) {
    for (int rnd : MATH_ROUNDING_DIRECTIONS_INCLUDING_UNKNOWN) {
      EXPECT_FP_EQ(aNaN, func(zero, rnd, 0U));
      EXPECT_FP_EQ(aNaN, func(neg_zero, rnd, 0U));

      EXPECT_FP_EQ(aNaN, func(inf, rnd, 0U));
      EXPECT_FP_EQ(aNaN, func(neg_inf, rnd, 0U));

      EXPECT_FP_EQ(aNaN, func(aNaN, rnd, 0U));
    }
  }

  void testRoundedNumbersWithinRange(FromfpxFunc func) {
    for (int rnd : MATH_ROUNDING_DIRECTIONS_INCLUDING_UNKNOWN) {
      EXPECT_FP_EQ(T(1.0), func(T(1.0), rnd, 2U));
      EXPECT_FP_EQ(T(-1.0), func(T(-1.0), rnd, 1U));
      EXPECT_FP_EQ(T(10.0), func(T(10.0), rnd, 5U));
      EXPECT_FP_EQ(T(-10.0), func(T(-10.0), rnd, 5U));
      EXPECT_FP_EQ(T(1234.0), func(T(1234.0), rnd, 12U));
      EXPECT_FP_EQ(T(-1234.0), func(T(-1234.0), rnd, 12U));
    }
  }

  void testRoundedNumbersOutsideRange(FromfpxFunc func) {
    for (int rnd : MATH_ROUNDING_DIRECTIONS_INCLUDING_UNKNOWN) {
      EXPECT_FP_EQ(aNaN, func(T(1.0), rnd, 1U));
      EXPECT_FP_EQ(aNaN, func(T(10.0), rnd, 4U));
      EXPECT_FP_EQ(aNaN, func(T(-10.0), rnd, 4U));
      EXPECT_FP_EQ(aNaN, func(T(1234.0), rnd, 11U));
      EXPECT_FP_EQ(aNaN, func(T(-1234.0), rnd, 11U));
    }
  }

  void testFractionsUpwardWithinRange(FromfpxFunc func) {
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1.0), func(T(0.5), FP_INT_UPWARD, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-0.0), func(T(-0.5), FP_INT_UPWARD, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1.0), func(T(0.115), FP_INT_UPWARD, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-0.0), func(T(-0.115), FP_INT_UPWARD, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1.0), func(T(0.715), FP_INT_UPWARD, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-0.0), func(T(-0.715), FP_INT_UPWARD, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(2.0), func(T(1.3), FP_INT_UPWARD, 3U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-1.0), func(T(-1.3), FP_INT_UPWARD, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(2.0), func(T(1.5), FP_INT_UPWARD, 3U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-1.0), func(T(-1.5), FP_INT_UPWARD, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(2.0), func(T(1.75), FP_INT_UPWARD, 3U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-1.0), func(T(-1.75), FP_INT_UPWARD, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(11.0), func(T(10.32), FP_INT_UPWARD, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-10.0), func(T(-10.32), FP_INT_UPWARD, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(11.0), func(T(10.65), FP_INT_UPWARD, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-10.0), func(T(-10.65), FP_INT_UPWARD, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1235.0), func(T(1234.38), FP_INT_UPWARD, 12U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1234.0), func(T(-1234.38), FP_INT_UPWARD, 12U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1235.0), func(T(1234.96), FP_INT_UPWARD, 12U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1234.0), func(T(-1234.96), FP_INT_UPWARD, 12U), FE_INEXACT);
  }

  void testFractionsUpwardOutsideRange(FromfpxFunc func) {
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

  void testFractionsDownwardWithinRange(FromfpxFunc func) {
    EXPECT_FP_EQ_WITH_EXCEPTION(T(0.0), func(T(0.5), FP_INT_DOWNWARD, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-1.0), func(T(-0.5), FP_INT_DOWNWARD, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(0.0), func(T(0.115), FP_INT_DOWNWARD, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-1.0), func(T(-0.115), FP_INT_DOWNWARD, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(0.0), func(T(0.715), FP_INT_DOWNWARD, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-1.0), func(T(-0.715), FP_INT_DOWNWARD, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1.0), func(T(1.3), FP_INT_DOWNWARD, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-2.0), func(T(-1.3), FP_INT_DOWNWARD, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1.0), func(T(1.5), FP_INT_DOWNWARD, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-2.0), func(T(-1.5), FP_INT_DOWNWARD, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1.0), func(T(1.75), FP_INT_DOWNWARD, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-2.0), func(T(-1.75), FP_INT_DOWNWARD, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(10.0), func(T(10.32), FP_INT_DOWNWARD, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-11.0), func(T(-10.32), FP_INT_DOWNWARD, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(10.0), func(T(10.65), FP_INT_DOWNWARD, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-11.0), func(T(-10.65), FP_INT_DOWNWARD, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1234.0), func(T(1234.38), FP_INT_DOWNWARD, 12U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1235.0), func(T(-1234.38), FP_INT_DOWNWARD, 12U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1234.0), func(T(1234.96), FP_INT_DOWNWARD, 12U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1235.0), func(T(-1234.96), FP_INT_DOWNWARD, 12U), FE_INEXACT);
  }

  void testFractionsDownwardOutsideRange(FromfpxFunc func) {
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

  void testFractionsTowardZeroWithinRange(FromfpxFunc func) {
    EXPECT_FP_EQ_WITH_EXCEPTION(T(0.0), func(T(0.5), FP_INT_TOWARDZERO, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-0.0), func(T(-0.5), FP_INT_TOWARDZERO, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(0.0), func(T(0.115), FP_INT_TOWARDZERO, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-0.0), func(T(-0.115), FP_INT_TOWARDZERO, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(0.0), func(T(0.715), FP_INT_TOWARDZERO, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-0.0), func(T(-0.715), FP_INT_TOWARDZERO, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1.0), func(T(1.3), FP_INT_TOWARDZERO, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-1.0), func(T(-1.3), FP_INT_TOWARDZERO, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1.0), func(T(1.5), FP_INT_TOWARDZERO, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-1.0), func(T(-1.5), FP_INT_TOWARDZERO, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1.0), func(T(1.75), FP_INT_TOWARDZERO, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-1.0), func(T(-1.75), FP_INT_TOWARDZERO, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(10.0), func(T(10.32), FP_INT_TOWARDZERO, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-10.0), func(T(-10.32), FP_INT_TOWARDZERO, 5U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(10.0), func(T(10.65), FP_INT_TOWARDZERO, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-10.0), func(T(-10.65), FP_INT_TOWARDZERO, 5U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1234.0), func(T(1234.38), FP_INT_TOWARDZERO, 12U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1234.0), func(T(-1234.38), FP_INT_TOWARDZERO, 12U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1234.0), func(T(1234.96), FP_INT_TOWARDZERO, 12U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1234.0), func(T(-1234.96), FP_INT_TOWARDZERO, 12U), FE_INEXACT);
  }

  void testFractionsTowardZeroOutsideRange(FromfpxFunc func) {
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

  void testFractionsToNearestFromZeroWithinRange(FromfpxFunc func) {
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1.0), func(T(0.5), FP_INT_TONEARESTFROMZERO, 2U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1.0), func(T(-0.5), FP_INT_TONEARESTFROMZERO, 1U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(0.0), func(T(0.115), FP_INT_TONEARESTFROMZERO, 1U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-0.0), func(T(-0.115), FP_INT_TONEARESTFROMZERO, 1U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1.0), func(T(0.715), FP_INT_TONEARESTFROMZERO, 2U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1.0), func(T(-0.715), FP_INT_TONEARESTFROMZERO, 1U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1.0), func(T(1.3), FP_INT_TONEARESTFROMZERO, 2U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1.0), func(T(-1.3), FP_INT_TONEARESTFROMZERO, 1U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(2.0), func(T(1.5), FP_INT_TONEARESTFROMZERO, 3U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-2.0), func(T(-1.5), FP_INT_TONEARESTFROMZERO, 2U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(2.0), func(T(1.75), FP_INT_TONEARESTFROMZERO, 3U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-2.0), func(T(-1.75), FP_INT_TONEARESTFROMZERO, 2U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(10.0), func(T(10.32), FP_INT_TONEARESTFROMZERO, 5U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-10.0), func(T(-10.32), FP_INT_TONEARESTFROMZERO, 5U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(11.0), func(T(10.65), FP_INT_TONEARESTFROMZERO, 5U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-11.0), func(T(-10.65), FP_INT_TONEARESTFROMZERO, 5U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1234.0), func(T(1234.38), FP_INT_TONEARESTFROMZERO, 12U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1234.0), func(T(-1234.38), FP_INT_TONEARESTFROMZERO, 12U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1235.0), func(T(1234.96), FP_INT_TONEARESTFROMZERO, 12U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1235.0), func(T(-1234.96), FP_INT_TONEARESTFROMZERO, 12U),
        FE_INEXACT);
  }

  void testFractionsToNearestFromZeroOutsideRange(FromfpxFunc func) {
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

  void testFractionsToNearestWithinRange(FromfpxFunc func) {
    EXPECT_FP_EQ_WITH_EXCEPTION(T(0.0), func(T(0.5), FP_INT_TONEAREST, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-0.0), func(T(-0.5), FP_INT_TONEAREST, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(0.0), func(T(0.115), FP_INT_TONEAREST, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-0.0), func(T(-0.115), FP_INT_TONEAREST, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1.0), func(T(0.715), FP_INT_TONEAREST, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-1.0), func(T(-0.715), FP_INT_TONEAREST, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1.0), func(T(1.3), FP_INT_TONEAREST, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-1.0), func(T(-1.3), FP_INT_TONEAREST, 1U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(2.0), func(T(1.5), FP_INT_TONEAREST, 3U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-2.0), func(T(-1.5), FP_INT_TONEAREST, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(2.0), func(T(1.75), FP_INT_TONEAREST, 3U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-2.0), func(T(-1.75), FP_INT_TONEAREST, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(10.0), func(T(10.32), FP_INT_TONEAREST, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-10.0), func(T(-10.32), FP_INT_TONEAREST, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(11.0), func(T(10.65), FP_INT_TONEAREST, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-11.0), func(T(-10.65), FP_INT_TONEAREST, 5U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1234.0), func(T(1234.38), FP_INT_TONEAREST, 12U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1234.0), func(T(-1234.38), FP_INT_TONEAREST, 12U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1235.0), func(T(1234.96), FP_INT_TONEAREST, 12U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1235.0), func(T(-1234.96), FP_INT_TONEAREST, 12U), FE_INEXACT);

    EXPECT_FP_EQ_WITH_EXCEPTION(T(2.0), func(T(2.3), FP_INT_TONEAREST, 3U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-2.0), func(T(-2.3), FP_INT_TONEAREST, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(2.0), func(T(2.5), FP_INT_TONEAREST, 3U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-2.0), func(T(-2.5), FP_INT_TONEAREST, 2U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(3.0), func(T(2.75), FP_INT_TONEAREST, 3U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-3.0), func(T(-2.75), FP_INT_TONEAREST, 3U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(5.0), func(T(5.3), FP_INT_TONEAREST, 4U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-5.0), func(T(-5.3), FP_INT_TONEAREST, 4U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(6.0), func(T(5.5), FP_INT_TONEAREST, 4U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-6.0), func(T(-5.5), FP_INT_TONEAREST, 4U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(6.0), func(T(5.75), FP_INT_TONEAREST, 4U),
                                FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-6.0), func(T(-5.75), FP_INT_TONEAREST, 4U),
                                FE_INEXACT);
  }

  void testFractionsToNearestOutsideRange(FromfpxFunc func) {
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

  void testFractionsToNearestFallbackWithinRange(FromfpxFunc func) {
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(0.0), func(T(0.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 1U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-0.0), func(T(-0.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 1U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(0.0), func(T(0.115), UNKNOWN_MATH_ROUNDING_DIRECTION, 1U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-0.0), func(T(-0.115), UNKNOWN_MATH_ROUNDING_DIRECTION, 1U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1.0), func(T(0.715), UNKNOWN_MATH_ROUNDING_DIRECTION, 2U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1.0), func(T(-0.715), UNKNOWN_MATH_ROUNDING_DIRECTION, 1U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1.0), func(T(1.3), UNKNOWN_MATH_ROUNDING_DIRECTION, 2U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1.0), func(T(-1.3), UNKNOWN_MATH_ROUNDING_DIRECTION, 1U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(2.0), func(T(1.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 3U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-2.0), func(T(-1.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 2U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(2.0), func(T(1.75), UNKNOWN_MATH_ROUNDING_DIRECTION, 3U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-2.0), func(T(-1.75), UNKNOWN_MATH_ROUNDING_DIRECTION, 2U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(10.0), func(T(10.32), UNKNOWN_MATH_ROUNDING_DIRECTION, 5U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-10.0), func(T(-10.32), UNKNOWN_MATH_ROUNDING_DIRECTION, 5U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(11.0), func(T(10.65), UNKNOWN_MATH_ROUNDING_DIRECTION, 5U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-11.0), func(T(-10.65), UNKNOWN_MATH_ROUNDING_DIRECTION, 5U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1234.0), func(T(1234.38), UNKNOWN_MATH_ROUNDING_DIRECTION, 12U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1234.0), func(T(-1234.38), UNKNOWN_MATH_ROUNDING_DIRECTION, 12U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(1235.0), func(T(1234.96), UNKNOWN_MATH_ROUNDING_DIRECTION, 12U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-1235.0), func(T(-1234.96), UNKNOWN_MATH_ROUNDING_DIRECTION, 12U),
        FE_INEXACT);

    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(2.0), func(T(2.3), UNKNOWN_MATH_ROUNDING_DIRECTION, 3U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-2.0), func(T(-2.3), UNKNOWN_MATH_ROUNDING_DIRECTION, 2U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(2.0), func(T(2.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 3U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-2.0), func(T(-2.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 2U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(3.0), func(T(2.75), UNKNOWN_MATH_ROUNDING_DIRECTION, 3U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-3.0), func(T(-2.75), UNKNOWN_MATH_ROUNDING_DIRECTION, 3U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(5.0), func(T(5.3), UNKNOWN_MATH_ROUNDING_DIRECTION, 4U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-5.0), func(T(-5.3), UNKNOWN_MATH_ROUNDING_DIRECTION, 4U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(6.0), func(T(5.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 4U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-6.0), func(T(-5.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 4U),
        FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(6.0), func(T(5.75), UNKNOWN_MATH_ROUNDING_DIRECTION, 4U), FE_INEXACT);
    EXPECT_FP_EQ_WITH_EXCEPTION(
        T(-6.0), func(T(-5.75), UNKNOWN_MATH_ROUNDING_DIRECTION, 4U),
        FE_INEXACT);
  }

  void testFractionsToNearestFallbackOutsideRange(FromfpxFunc func) {
    EXPECT_FP_EQ(aNaN, func(T(0.715), UNKNOWN_MATH_ROUNDING_DIRECTION, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.3), UNKNOWN_MATH_ROUNDING_DIRECTION, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 2U));
    EXPECT_FP_EQ(aNaN, func(T(-1.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 1U));
    EXPECT_FP_EQ(aNaN, func(T(1.75), UNKNOWN_MATH_ROUNDING_DIRECTION, 2U));
    EXPECT_FP_EQ(aNaN, func(T(-1.75), UNKNOWN_MATH_ROUNDING_DIRECTION, 1U));
    EXPECT_FP_EQ(aNaN, func(T(10.32), UNKNOWN_MATH_ROUNDING_DIRECTION, 4U));
    EXPECT_FP_EQ(aNaN, func(T(-10.32), UNKNOWN_MATH_ROUNDING_DIRECTION, 4U));
    EXPECT_FP_EQ(aNaN, func(T(10.65), UNKNOWN_MATH_ROUNDING_DIRECTION, 4U));
    EXPECT_FP_EQ(aNaN, func(T(-10.65), UNKNOWN_MATH_ROUNDING_DIRECTION, 4U));
    EXPECT_FP_EQ(aNaN, func(T(1234.38), UNKNOWN_MATH_ROUNDING_DIRECTION, 11U));
    EXPECT_FP_EQ(aNaN, func(T(-1234.38), UNKNOWN_MATH_ROUNDING_DIRECTION, 11U));
    EXPECT_FP_EQ(aNaN, func(T(1234.96), UNKNOWN_MATH_ROUNDING_DIRECTION, 11U));
    EXPECT_FP_EQ(aNaN, func(T(-1234.96), UNKNOWN_MATH_ROUNDING_DIRECTION, 11U));

    EXPECT_FP_EQ(aNaN, func(T(2.3), UNKNOWN_MATH_ROUNDING_DIRECTION, 2U));
    EXPECT_FP_EQ(aNaN, func(T(-2.3), UNKNOWN_MATH_ROUNDING_DIRECTION, 1U));
    EXPECT_FP_EQ(aNaN, func(T(2.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 2U));
    EXPECT_FP_EQ(aNaN, func(T(-2.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 1U));
    EXPECT_FP_EQ(aNaN, func(T(2.75), UNKNOWN_MATH_ROUNDING_DIRECTION, 2U));
    EXPECT_FP_EQ(aNaN, func(T(-2.75), UNKNOWN_MATH_ROUNDING_DIRECTION, 2U));
    EXPECT_FP_EQ(aNaN, func(T(5.3), UNKNOWN_MATH_ROUNDING_DIRECTION, 3U));
    EXPECT_FP_EQ(aNaN, func(T(-5.3), UNKNOWN_MATH_ROUNDING_DIRECTION, 3U));
    EXPECT_FP_EQ(aNaN, func(T(5.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 3U));
    EXPECT_FP_EQ(aNaN, func(T(-5.5), UNKNOWN_MATH_ROUNDING_DIRECTION, 3U));
    EXPECT_FP_EQ(aNaN, func(T(5.75), UNKNOWN_MATH_ROUNDING_DIRECTION, 3U));
    EXPECT_FP_EQ(aNaN, func(T(-5.75), UNKNOWN_MATH_ROUNDING_DIRECTION, 3U));
  }
};

#define LIST_FROMFPX_TESTS(T, func)                                            \
  using LlvmLibcFromfpxTest = FromfpxTestTemplate<T>;                          \
  TEST_F(LlvmLibcFromfpxTest, SpecialNumbersNonzeroWidth) {                    \
    testSpecialNumbersNonzeroWidth(&func);                                     \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, SpecialNumbersZeroWidth) {                       \
    testSpecialNumbersZeroWidth(&func);                                        \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, RoundedNumbersWithinRange) {                     \
    testRoundedNumbersWithinRange(&func);                                      \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, RoundedNumbersOutsideRange) {                    \
    testRoundedNumbersOutsideRange(&func);                                     \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, FractionsUpwardWithinRange) {                    \
    testFractionsUpwardWithinRange(&func);                                     \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, FractionsUpwardOutsideRange) {                   \
    testFractionsUpwardOutsideRange(&func);                                    \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, FractionsDownwardWithinRange) {                  \
    testFractionsDownwardWithinRange(&func);                                   \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, FractionsDownwardOutsideRange) {                 \
    testFractionsDownwardOutsideRange(&func);                                  \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, FractionsTowardZeroWithinRange) {                \
    testFractionsTowardZeroWithinRange(&func);                                 \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, FractionsTowardZeroOutsideRange) {               \
    testFractionsTowardZeroOutsideRange(&func);                                \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, FractionsToNearestFromZeroWithinRange) {         \
    testFractionsToNearestFromZeroWithinRange(&func);                          \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, FractionsToNearestFromZeroOutsideRange) {        \
    testFractionsToNearestFromZeroOutsideRange(&func);                         \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, FractionsToNearestWithinRange) {                 \
    testFractionsToNearestWithinRange(&func);                                  \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, FractionsToNearestOutsideRange) {                \
    testFractionsToNearestOutsideRange(&func);                                 \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, FractionsToNearestFallbackWithinRange) {         \
    testFractionsToNearestFallbackWithinRange(&func);                          \
  }                                                                            \
  TEST_F(LlvmLibcFromfpxTest, FractionsToNearestFallbackOutsideRange) {        \
    testFractionsToNearestFallbackOutsideRange(&func);                         \
  }

#endif // LIBC_TEST_SRC_MATH_SMOKE_FROMFPXTEST_H
