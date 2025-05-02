//===-- Utility class to test bitsfx functions ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "src/__support/fixed_point/fx_rep.h"
#include "src/__support/fixed_point/divifx.h"

template <typename T, typename XType>
class DiviFxTest : public LIBC_NAMESPACE::testing::Test {

  using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<T>;
  static constexpr T zero = FXRep::ZERO();
  static constexpr T max = FXRep::MAX();
  static constexpr T min = FXRep::MIN();
  static constexpr T one_half = FXRep::ONE_HALF();
  static constexpr T one_fourth = FXRep::ONE_FOURTH();
  static constexpr T eps = FXRep::EPS();

public:
  typedef XType (*DiviFxFunc)(int, T);

  void testSpecialNumbers(DiviFxFunc func) {
    EXPECT_EQ(static_cast<XType>(200), func(100, one_half));
    EXPECT_EQ(static_cast<XType>(400), func(100, one_fourth));
    // std::cout << one_half << " " << one_fourth << std::endl;
  }
};

#define LIST_DIVIFX_TESTS(Name, T, XType, func)                                \
  using LlvmLibcDivifx##Name##Test = DiviFxTest<T, XType>;                       \
  TEST_F(LlvmLibcDivifx##Name##Test, SpecialNumbers) {                           \
    testSpecialNumbers(&func);                                                 \
  }                                                                            \
  static_assert(true, "Require semicolon.")
