//===-- Utility class to test divifx functions ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "hdr/signal_macros.h"
#include "llvm-libc-macros/stdfix-macros.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/fixed_point/fx_rep.h"

namespace cpp = LIBC_NAMESPACE::cpp;

template <typename IntType, typename FXType>
class DiviFxTest : public LIBC_NAMESPACE::testing::Test {

  using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<FXType>;

  static constexpr FXType max = FXRep::MAX();
  static constexpr FXType min = FXRep::MIN();
  static constexpr FXType zero = FXRep::ZERO();
  static constexpr FXType epsilon = FXRep::EPS();
  static constexpr FXType one_half = FXRep::ONE_HALF();
  static constexpr FXType one_fourth = FXRep::ONE_FOURTH();
  static constexpr FXType one_eighth = FXRep::ONE_EIGHTH();

  static constexpr bool is_signed = (FXRep::SIGN_LEN > 0);
  static constexpr bool has_integral = (FXRep::INTEGRAL_LEN > 0);

public:
  typedef IntType (*DiviFxFunc)(IntType, FXType);

  void testBasicNumbers(DiviFxFunc func) {

    EXPECT_EQ(func(0, one_eighth), static_cast<IntType>(0));
    EXPECT_EQ(func(0, one_fourth), static_cast<IntType>(0));
    EXPECT_EQ(func(0, one_half), static_cast<IntType>(0));
    EXPECT_EQ(func(1, one_eighth), static_cast<IntType>(8));
    EXPECT_EQ(func(1, one_fourth), static_cast<IntType>(4));
    EXPECT_EQ(func(1, one_half), static_cast<IntType>(2));
    EXPECT_EQ(func(2, one_eighth), static_cast<IntType>(16));
    EXPECT_EQ(func(2, one_fourth), static_cast<IntType>(8));
    EXPECT_EQ(func(2, one_half), static_cast<IntType>(4));

    // Verify rounding towards 0.
    EXPECT_EQ(func(1, 3 * one_fourth), static_cast<IntType>(1));
    EXPECT_EQ(func(2, 3 * one_fourth), static_cast<IntType>(2));

    if constexpr (is_signed) {
      EXPECT_EQ(func(-1, one_half), static_cast<IntType>(-2));
      EXPECT_EQ(func(1, -one_half), static_cast<IntType>(-2));
      EXPECT_EQ(func(-1, -one_half), static_cast<IntType>(2));
      EXPECT_EQ(func(-2, one_fourth), static_cast<IntType>(-8));
      EXPECT_EQ(func(2, -one_fourth), static_cast<IntType>(-8));

      // Verify rounding towards 0.
      EXPECT_EQ(func(-1, 3 * one_fourth), static_cast<IntType>(-1));
      EXPECT_EQ(func(1, -3 * one_fourth), static_cast<IntType>(-1));
      EXPECT_EQ(func(-2, 3 * one_fourth), static_cast<IntType>(-2));
      EXPECT_EQ(func(2, -3 * one_fourth), static_cast<IntType>(-2));

      EXPECT_EQ(func(1, static_cast<FXType>(-1)), static_cast<IntType>(-1));
      EXPECT_EQ(func(-1, static_cast<FXType>(-1)), static_cast<IntType>(1));
    }

    if constexpr (has_integral) {
      EXPECT_EQ(func(1, static_cast<FXType>(1)), static_cast<IntType>(1));

      if constexpr (is_signed) {
        EXPECT_EQ(func(-1, static_cast<FXType>(1)), static_cast<IntType>(-1));
      }

      // Only run these tests for accum types that can represent these operands.
      constexpr FXType max_test_operand = static_cast<FXType>(4);

      if constexpr (max >= max_test_operand) {
        EXPECT_EQ(func(3, 2.5), static_cast<IntType>(1));
        EXPECT_EQ(func(2, 3.5), static_cast<IntType>(0));
        EXPECT_EQ(func(3, 1.5), static_cast<IntType>(2));
        EXPECT_EQ(func(4, 2.0), static_cast<IntType>(2));

        if constexpr (is_signed) {
          EXPECT_EQ(func(2, -3.5), static_cast<IntType>(0));
          EXPECT_EQ(func(-3, 2.5), static_cast<IntType>(-1));
          EXPECT_EQ(func(-3, 1.5), static_cast<IntType>(-2));
          EXPECT_EQ(func(-3, -2.5), static_cast<IntType>(1));
        }
      }
    }
  }

  void testEdgeCases(DiviFxFunc func) {

    constexpr IntType int_max = cpp::numeric_limits<IntType>::max();

    if constexpr (has_integral) {
      EXPECT_EQ(func(int_max, static_cast<FXType>(1)), int_max);
    }

    if constexpr (is_signed) {
      EXPECT_EQ(func(int_max, static_cast<FXType>(-1)), -int_max);
    }

    // Only valid when integral bits (excluding sign) exceed fraction bits.
    // Otherwise 1<<FRACTION_LEN may not fit in IntType.
    if constexpr ((FXRep::INTEGRAL_LEN) > FXRep::FRACTION_LEN) {
      constexpr IntType epsilon_result = static_cast<IntType>(1)
                                         << FXRep::FRACTION_LEN;
      EXPECT_EQ(func(1, epsilon), epsilon_result);
      if constexpr (is_signed) {
        EXPECT_EQ(func(1, -epsilon), -epsilon_result);
        EXPECT_EQ(func(-1, epsilon), -epsilon_result);
        EXPECT_EQ(func(-1, -epsilon), epsilon_result);
      }
    }

    if constexpr (has_integral) {
      constexpr IntType largest_positive =
          ((static_cast<IntType>(1) << FXRep::INTEGRAL_LEN) - 1);
      EXPECT_EQ(func(largest_positive, static_cast<FXType>(largest_positive)),
                static_cast<IntType>(1));

      constexpr IntType largest_power_of_two = static_cast<IntType>(1)
                                               << (FXRep::INTEGRAL_LEN - 1);
      EXPECT_EQ(
          func(largest_power_of_two, static_cast<FXType>(largest_power_of_two)),
          static_cast<IntType>(1));
    }

    if constexpr (!has_integral) {
      EXPECT_EQ(func(1, max), static_cast<IntType>(1));

      if constexpr (is_signed) {
        EXPECT_EQ(func(1, min), static_cast<IntType>(-1));
      }
    }

    if constexpr (has_integral) {
      EXPECT_EQ(func(1, max), static_cast<IntType>(0));

      if constexpr (is_signed) {
        EXPECT_EQ(func(1, min), static_cast<IntType>(0));
      }
    }
  }

  void testInvalidNumbers(DiviFxFunc func) {

    EXPECT_DEATH([func] { func(1, zero); }, WITH_SIGNAL(-1));
    if constexpr (has_integral) {
      EXPECT_DEATH([func] { func(2, zero); }, WITH_SIGNAL(-1));
    }
  }
};

#if defined(LIBC_ADD_NULL_CHECKS)
#define LIST_DIVIFX_TESTS(Name, IntType, FXType, func)                         \
  using LlvmLibcDivi##Name##Test = DiviFxTest<IntType, FXType>;                \
  TEST_F(LlvmLibcDivi##Name##Test, InvalidNumbers) {                           \
    testInvalidNumbers(&func);                                                 \
  }                                                                            \
  TEST_F(LlvmLibcDivi##Name##Test, BasicNumbers) { testBasicNumbers(&func); }  \
  TEST_F(LlvmLibcDivi##Name##Test, EdgeCases) { testEdgeCases(&func); }        \
  static_assert(true, "Require semicolon.")
#else
#define LIST_DIVIFX_TESTS(Name, IntType, FXType, func)                         \
  using LlvmLibcDivi##Name##Test = DiviFxTest<IntType, FXType>;                \
  TEST_F(LlvmLibcDivi##Name##Test, BasicNumbers) { testBasicNumbers(&func); }  \
  TEST_F(LlvmLibcDivi##Name##Test, EdgeCases) { testEdgeCases(&func); }        \
  static_assert(true, "Require semicolon.")
#endif // LIBC_ADD_NULL_CHECKS
