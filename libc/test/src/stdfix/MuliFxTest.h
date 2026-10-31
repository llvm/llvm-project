//===-- Utility class to test mulifx functions ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "llvm-libc-macros/stdfix-macros.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/fixed_point/fx_rep.h"

namespace cpp = LIBC_NAMESPACE::cpp;

template <typename IntType, typename FXType>
class MuliFxTest : public LIBC_NAMESPACE::testing::Test {

  using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<FXType>;

  static constexpr FXType max = FXRep::MAX();
  static constexpr FXType min = FXRep::MIN();
  static constexpr FXType epsilon = FXRep::EPS();
  static constexpr FXType one_half = FXRep::ONE_HALF();
  static constexpr FXType one_fourth = FXRep::ONE_FOURTH();
  static constexpr FXType one_eighth = FXRep::ONE_EIGHTH();

  static constexpr bool is_signed = (FXRep::SIGN_LEN > 0);
  static constexpr bool has_integral = (FXRep::INTEGRAL_LEN > 0);

public:
  typedef IntType (*MuliFxFunc)(IntType, FXType);

  void testBasicNumbers(MuliFxFunc func) {

    EXPECT_EQ(func(0, epsilon), static_cast<IntType>(0));
    EXPECT_EQ(func(0, max), static_cast<IntType>(0));

    EXPECT_EQ(func(1, one_eighth), static_cast<IntType>(0));
    EXPECT_EQ(func(7, one_eighth), static_cast<IntType>(0));
    EXPECT_EQ(func(8, one_eighth), static_cast<IntType>(1));
    EXPECT_EQ(func(9, one_eighth), static_cast<IntType>(1));
    EXPECT_EQ(func(16, one_eighth), static_cast<IntType>(2));
    EXPECT_EQ(func(4, one_fourth), static_cast<IntType>(1));
    EXPECT_EQ(func(8, one_fourth), static_cast<IntType>(2));
    EXPECT_EQ(func(1, one_half), static_cast<IntType>(0));
    EXPECT_EQ(func(3, one_half), static_cast<IntType>(1));
    EXPECT_EQ(func(4, one_half), static_cast<IntType>(2));

    // Verify rounding towards 0.
    EXPECT_EQ(func(1, 3 * one_fourth), static_cast<IntType>(0));
    EXPECT_EQ(func(2, 3 * one_fourth), static_cast<IntType>(1));
    EXPECT_EQ(func(3, 3 * one_fourth), static_cast<IntType>(2));

    if constexpr (is_signed) {
      EXPECT_EQ(func(-1, one_half), static_cast<IntType>(0));
      EXPECT_EQ(func(-3, one_half), static_cast<IntType>(-1));
      EXPECT_EQ(func(3, -one_half), static_cast<IntType>(-1));
      EXPECT_EQ(func(-3, -one_half), static_cast<IntType>(1));
      EXPECT_EQ(func(-8, one_eighth), static_cast<IntType>(-1));
      EXPECT_EQ(func(-9, one_eighth), static_cast<IntType>(-1));
      EXPECT_EQ(func(-1, 3 * one_fourth), static_cast<IntType>(0));
      EXPECT_EQ(func(-2, 3 * one_fourth), static_cast<IntType>(-1));
      EXPECT_EQ(func(-3, 3 * one_fourth), static_cast<IntType>(-2));
    }

    if constexpr (has_integral) {
      EXPECT_EQ(func(1, static_cast<FXType>(1)), static_cast<IntType>(1));
      EXPECT_EQ(func(2, static_cast<FXType>(2)), static_cast<IntType>(4));

      if constexpr (is_signed) {
        EXPECT_EQ(func(1, static_cast<FXType>(-1)), static_cast<IntType>(-1));
        EXPECT_EQ(func(-1, static_cast<FXType>(1)), static_cast<IntType>(-1));
        EXPECT_EQ(func(-1, static_cast<FXType>(-1)), static_cast<IntType>(1));
      }

      // Only run these tests for accum types that can represent these operands.
      constexpr FXType max_test_operand = static_cast<FXType>(4);

      if constexpr (max >= max_test_operand) {
        EXPECT_EQ(func(3, 2.5), static_cast<IntType>(7));
        EXPECT_EQ(func(4, 1.5), static_cast<IntType>(6));

        if constexpr (is_signed) {
          EXPECT_EQ(func(-3, 2.5), static_cast<IntType>(-7));
          EXPECT_EQ(func(-3, -2.5), static_cast<IntType>(7));
        }
      }
    }
  }

  void testEdgeCases(MuliFxFunc func) {

    constexpr IntType int_max = cpp::numeric_limits<IntType>::max();
    constexpr IntType int_min = cpp::numeric_limits<IntType>::min();

    if constexpr (has_integral) {
      EXPECT_EQ(func(int_max, static_cast<FXType>(1)), int_max);

      if constexpr (is_signed) {
        EXPECT_EQ(func(int_max, static_cast<FXType>(-1)), -int_max);
      }
    }

    EXPECT_EQ(func(int_max, one_half), int_max / 2);

    if constexpr (is_signed) {
      EXPECT_EQ(func(int_min, one_half), int_min / 2);
    }

    if constexpr (FXRep::FRACTION_LEN < cpp::numeric_limits<IntType>::digits) {
      constexpr IntType epsilon_reciprocal = static_cast<IntType>(1)
                                             << FXRep::FRACTION_LEN;
      EXPECT_EQ(func(epsilon_reciprocal, epsilon), static_cast<IntType>(1));

      if constexpr (is_signed) {
        EXPECT_EQ(func(-epsilon_reciprocal, epsilon), static_cast<IntType>(-1));
        EXPECT_EQ(func(epsilon_reciprocal, -epsilon), static_cast<IntType>(-1));
        EXPECT_EQ(func(-epsilon_reciprocal, -epsilon), static_cast<IntType>(1));
      }
    }

    if constexpr (has_integral &&
                  FXRep::INTEGRAL_LEN < cpp::numeric_limits<IntType>::digits) {
      constexpr IntType integral_max =
          (static_cast<IntType>(1) << FXRep::INTEGRAL_LEN) - 1;

      EXPECT_EQ(func(1, max), integral_max);

      if constexpr (is_signed) {
        EXPECT_EQ(func(1, min), -static_cast<IntType>(integral_max + 1));
        EXPECT_EQ(func(-1, max), -integral_max);
      }
    }

    if constexpr (!has_integral) {
      EXPECT_EQ(func(1, max), static_cast<IntType>(0));
      EXPECT_EQ(func(2, max), static_cast<IntType>(1));

      if constexpr (is_signed) {
        EXPECT_EQ(func(1, min), static_cast<IntType>(-1));
        EXPECT_EQ(func(-1, max), static_cast<IntType>(0));
        EXPECT_EQ(func(-1, min), static_cast<IntType>(1));
      }
    }
  }
};

#define LIST_MULIFX_TESTS(Name, IntType, FXType, func)                         \
  using LlvmLibcMuli##Name##Test = MuliFxTest<IntType, FXType>;                \
  TEST_F(LlvmLibcMuli##Name##Test, BasicNumbers) { testBasicNumbers(&func); }  \
  TEST_F(LlvmLibcMuli##Name##Test, EdgeCases) { testEdgeCases(&func); }        \
  static_assert(true, "Require semicolon.")
