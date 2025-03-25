//===-- Utility class to test idivfx functions ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "src/__support/fixed_point/fx_rep.h"

template <typename T, typename XType>
class IdivTest : public LIBC_NAMESPACE::testing::Test {

  using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<T>;

  static constexpr T zero = FXRep::ZERO();
  static constexpr T max = FXRep::MAX();
  static constexpr T min = FXRep::MIN();
  static constexpr T one_half = FXRep::ONE_HALF();
  static constexpr T one_fourth = FXRep::ONE_FOURTH();
  static constexpr T eps = FXRep::EPS();

public:
  typedef XType (*IdivFunc)(T, T);

  void testSpecialNumbers(IdivFunc func) {
    EXPECT_EQ(func(one_half, one_fourth), 2);
  }
};

#define LIST_IDIV_TESTS(Name, T, XType, func)                                  \
  using LlvmLibcIdiv##Name##Test = IdivTest<T, XType>;                         \
  TEST_F(LlvmLibcIdiv##Name##Test, SpecialNumbers) {                           \
    testSpecialNumbers(&func);                                                 \
  }                                                                            \
  static_assert(true, "Require semicolon.")
