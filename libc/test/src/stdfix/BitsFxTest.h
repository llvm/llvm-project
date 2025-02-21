//===-- Utility class to test bitsfx functions ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "src/__support/fixed_point/fx_rep.h"

template <typename From, typename To>
class BitsFxTest : public LIBC_NAMESPACE::testing::Test {

  using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<From>;
  static constexpr From zero = FXRep::ZERO();
  static constexpr From max = FXRep::MAX();
  static constexpr From min = FXRep::MIN();
  static constexpr From one_half = FXRep::ONE_HALF();
  static constexpr From one_fourth = FXRep::ONE_FOURTH();
  static constexpr From eps = FXRep::EPS();
  // (0.42)_10 =
  // (0.0110101110000101000111101011100001010001111010111000010100011110)_2 =
  // (0.0x6b851eb851eb851e)_16
  static constexpr unsigned long long zero_point_forty_two =
      0x6b851eb851eb851eull;

  static constexpr unsigned long long maxval = ~(1ULL << FXRep::VALUE_LEN);
  static constexpr unsigned long long minval = -(maxval + 1ULL);

public:
  typedef To (*BitsFxFunc)(From);

  void testSpecialNumbers(BitsFxFunc func) {
    EXPECT_EQ(static_cast<To>(0), func(zero));
    EXPECT_EQ(static_cast<To>(1 << (FXRep::FRACTION_LEN - 1)), func(one_half));
    EXPECT_EQ(static_cast<To>(1 << (FXRep::FRACTION_LEN - 2)),
              func(one_fourth));
    EXPECT_EQ(static_cast<To>(1), func(eps));
    EXPECT_EQ(static_cast<To>(maxval), func(max));
    EXPECT_EQ(static_cast<To>(minval), func(min));

    // (0.6875)_10 = (0.1011)_2
    EXPECT_EQ(static_cast<To>(11 << (FXRep::FRACTION_LEN - 4)), func(0.6875));

    EXPECT_EQ(
        static_cast<To>(zero_point_forty_two >> (64 - FXRep::FRACTION_LEN)),
        func(0.42));

    // EXPECT_EQ(static_cast<To>(0), func(5));

    // if constexpr (static_cast<int>(min) <= -16)
    //   EXPECT_EQ(static_cast<To>(0), func(-16));

    // if constexpr (static_cast<int>(min) <= -10)
    //   EXPECT_EQ(static_cast<To>(0), func(-10));

    // EXPECT_EQ(static_cast<To>(), func(max));

    // if constexpr (static_cast<int>(max) >= 16.25)
    //   EXPECT_EQ(static_cast<To>(0), func(16.25));

    // if constexpr (static_cast<int>(max) > -10)
    //   EXPECT_EQ(static_cast<To>(0), func(16));
  }
};

#define LIST_BITSFX_TESTS(From, To, func)                                      \
  using LlvmLibcBitsFxTest = BitsFxTest<From, To>;                             \
  TEST_F(LlvmLibcBitsFxTest, SpecialNumbers) { testSpecialNumbers(&func); }    \
  static_assert(true, "Require semicolon.")
