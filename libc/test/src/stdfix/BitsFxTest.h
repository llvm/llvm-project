//===-- Utility class to test fixed point to int conversions ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "include/llvm-libc-types/stdfix-types.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/fixed_point/fx_bits.h"

template <typename T, typename XType>
class BitsFxTest : public LIBC_NAMESPACE::testing::Test {
  using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<T>;
  static constexpr T zero = FXRep::ZERO();
  static constexpr T min = FXRep::MIN();
  static constexpr T max = FXRep::MAX();
  static constexpr T half = static_cast<T>(0.5);
  static constexpr T quarter = static_cast<T>(0.25);
  static constexpr T one =
      (FXRep::INTEGRAL_LEN > 0) ? static_cast<T>(1) : FXRep::MAX();
  static constexpr T eps = FXRep::EPS();
  constexpr XType get_one_or_saturated_fraction() {
    if (FXRep::INTEGRAL_LEN > 0) {
      return static_cast<XType>(static_cast<XType>(0x1) << FXRep::FRACTION_LEN);
    } else {
      return static_cast<XType>(
          LIBC_NAMESPACE::mask_trailing_ones<typename FXRep::StorageType,
                                             FXRep::FRACTION_LEN>());
    }
  }

public:
  typedef XType (*BitsFxFunc)(T);

  void test_special_numbers(BitsFxFunc func) {
    EXPECT_EQ(static_cast<XType>(0), func(zero));
    EXPECT_EQ(static_cast<XType>(0x1), func(eps));
    EXPECT_EQ(static_cast<XType>(static_cast<XType>(0x1)
                                 << (FXRep::FRACTION_LEN - 1)),
              func(half));
    // Occupy the bit to the left of the fixed point for Accum types
    // Saturate fraction portion for Fract types
    EXPECT_EQ(get_one_or_saturated_fraction(), func(one));
  }
};

#define LIST_BITSFX_TEST(Name, T, XType, func)                                 \
  using LlvmLibcBits##Name##Test = BitsFxTest<T, XType>;                       \
  TEST_F(LlvmLibcBits##Name##Test, SpecialNumbers) {                           \
    test_special_numbers(&func);                                               \
  }                                                                            \
  static_assert(true, "Require semicolon.")
