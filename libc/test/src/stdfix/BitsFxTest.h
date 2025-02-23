//===-- Utility class to test bitsfx functions ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "src/__support/fixed_point/fx_rep.h"

template <typename T, typename XType>
class BitsFxTest : public LIBC_NAMESPACE::testing::Test {

  using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<T>;
  static constexpr T zero = FXRep::ZERO();
  static constexpr T max = FXRep::MAX();
  static constexpr T min = FXRep::MIN();
  static constexpr T one_half = FXRep::ONE_HALF();
  static constexpr T one_fourth = FXRep::ONE_FOURTH();
  static constexpr T eps = FXRep::EPS();

  // (0.42)_10 =
  // (0.0110101110000101000111101011100001010001111010111000010100011110)_2 =
  // (0.0x6b851eb851eb851e)_16
  static constexpr unsigned long long zero_point_forty_two =
      0x6b851eb851eb851eULL;

public:
  typedef XType (*BitsFxFunc)(T);

  void testSpecialNumbers(BitsFxFunc func) {
    EXPECT_EQ(static_cast<XType>(0), func(zero));
    EXPECT_EQ(static_cast<XType>(1ULL << (FXRep::FRACTION_LEN - 1)),
              func(one_half));
    EXPECT_EQ(static_cast<XType>(1ULL << (FXRep::FRACTION_LEN - 2)),
              func(one_fourth));
    EXPECT_EQ(static_cast<XType>(1), func(eps));

    // (0.6875)_10 = (0.1011)_2
    EXPECT_EQ(static_cast<XType>(11ULL << (FXRep::FRACTION_LEN - 4)),
              func(0.6875));

    EXPECT_EQ(
        static_cast<XType>(zero_point_forty_two >> (64 - FXRep::FRACTION_LEN)),
        func(0.42));
  }
};

#define LIST_BITSFX_TESTS(Name, T, XType, func)                                \
  using LlvmLibcBits##Name##Test = BitsFxTest<T, XType>;                       \
  TEST_F(LlvmLibcBits##Name##Test, SpecialNumbers) {                           \
    testSpecialNumbers(&func);                                                 \
  }                                                                            \
  static_assert(true, "Require semicolon.")
