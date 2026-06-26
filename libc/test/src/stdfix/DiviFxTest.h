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
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/fixed_point/fx_rep.h"

template <typename IntType, typename FXType>
class DiviFxTest : public LIBC_NAMESPACE::testing::Test {

  using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<FXType>;

  static constexpr FXType zero = FXRep::ZERO();
  static constexpr FXType min = FXRep::MIN();
  static constexpr FXType one_half = FXRep::ONE_HALF();
  static constexpr FXType one_fourth = FXRep::ONE_FOURTH();

public:
  typedef IntType (*DiviFxFunc)(IntType, FXType);

  void testBasicNumbers(DiviFxFunc func) {
    constexpr bool is_signed = (FXRep::SIGN_LEN > 0);

    EXPECT_EQ(func(1, one_fourth), static_cast<IntType>(4));
    EXPECT_EQ(func(1, one_half), static_cast<IntType>(2));
    EXPECT_EQ(func(1, 0.125r), static_cast<IntType>(8));
    EXPECT_EQ(func(2, one_fourth), static_cast<IntType>(8));
    EXPECT_EQ(func(2, one_half), static_cast<IntType>(4));
    EXPECT_EQ(func(2, 0.125r), static_cast<IntType>(16));

    // verify rounding towards 0
    EXPECT_EQ(func(1, 3 * one_fourth), static_cast<IntType>(1));
    EXPECT_EQ(func(2, 3 * one_fourth), static_cast<IntType>(2));

    if constexpr (is_signed) {
      EXPECT_EQ(func(-1, one_half), static_cast<IntType>(-2));
      EXPECT_EQ(func(1, -one_half), static_cast<IntType>(-2));
      EXPECT_EQ(func(-1, -one_half), static_cast<IntType>(2));
      EXPECT_EQ(func(-2, one_fourth), static_cast<IntType>(-8));
      EXPECT_EQ(func(2, -one_fourth), static_cast<IntType>(-8));

      // verify rounding towards 0
      EXPECT_EQ(func(-1, 3 * one_fourth), static_cast<IntType>(-1));
      EXPECT_EQ(func(1, -3 * one_fourth), static_cast<IntType>(-1));
      EXPECT_EQ(func(-2, 3 * one_fourth), static_cast<IntType>(-2));
      EXPECT_EQ(func(2, -3 * one_fourth), static_cast<IntType>(-2));
    } else {
      // min of unsigned type = 0
      EXPECT_EQ(func(min, one_half), static_cast<IntType>(0));
      EXPECT_EQ(func(min, one_fourth), static_cast<IntType>(0));
    }
  }

  void testInvalidNumbers(DiviFxFunc func) {
    EXPECT_DEATH([func] { func(1, zero); }, WITH_SIGNAL(-1));
  }
};

#if defined(LIBC_ADD_NULL_CHECKS)
#define LIST_DIVIFX_TESTS(Name, IntType, FXType, func)                         \
  using LlvmLibcDivi##Name##Test = DiviFxTest<IntType, FXType>;                \
  TEST_F(LlvmLibcDivi##Name##Test, InvalidNumbers) {                           \
    testInvalidNumbers(&func);                                                 \
  }                                                                            \
  TEST_F(LlvmLibcDivi##Name##Test, BasicNumbers) { testBasicNumbers(&func); }  \
  static_assert(true, "Require semicolon.")
#else
#define LIST_DIVIFX_TESTS(Name, IntType, FXType, func)                         \
  using LlvmLibcDivi##Name##Test = DiviFxTest<IntType, FXType>;                \
  TEST_F(LlvmLibcDivi##Name##Test, BasicNumbers) { testBasicNumbers(&func); }  \
  static_assert(true, "Require semicolon.")
#endif // LIBC_ADD_NULL_CHECKS
