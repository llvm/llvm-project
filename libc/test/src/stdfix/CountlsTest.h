//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "src/__support/fixed_point/fx_rep.h"

template <typename T> class CountlsTest : public LIBC_NAMESPACE::testing::Test {

  using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<T>;
  static constexpr T zero = FXRep::ZERO();
  static constexpr T max = FXRep::MAX();
  static constexpr T min = FXRep::MIN();
  static constexpr T one_half = FXRep::ONE_HALF();
  static constexpr T one_fourth = FXRep::ONE_FOURTH();
  static constexpr T eps = FXRep::EPS();

  static constexpr auto value_len = LIBC_NAMESPACE::fixed_point::get_value_len<FXRep>();

public:
  typedef int (*CountlsFunc)(T);

  void testSpecialNumbers(CountlsFunc func) {
    constexpr bool is_signed = (min != zero);

    EXPECT_EQ(FXRep::INTEGRAL_LEN, func(one_half));
    EXPECT_EQ(FXRep::INTEGRAL_LEN + 1, func(one_fourth));
    EXPECT_EQ(value_len, func(zero));
    EXPECT_EQ(value_len - 1, func(eps));
    EXPECT_EQ(0, func(max));
    // If signed, left shifting the minimum value will overflow, so countls = 0.
    // If unsigned, the minimum value is zero, so countls is the number of value
    // bits according to ISO/IEC TR 18037.
    EXPECT_EQ(is_signed ? 0 : value_len, func(min));

    if constexpr (is_signed) {
      EXPECT_EQ(value_len, func(-eps));
    }
  }
};

#define LIST_COUNTLS_TESTS(T, func)                                            \
  using LlvmLibcCountlsTest = CountlsTest<T>;                                  \
  TEST_F(LlvmLibcCountlsTest, SpecialNumbers) { testSpecialNumbers(&func); }   \
  static_assert(true, "Require semicolon.")
