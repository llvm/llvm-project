//===-- stl_extras_test.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime.
//
// Note:
//   This unit test was adapted from tests in
//   llvm/unittests/ADT/STLExtrasTest.cpp
//
//===----------------------------------------------------------------------===//

#include "stl_extras.h"
#include "gtest/gtest.h"

using namespace __orc_rt;

TEST(STLExtrasTest, ApplyTuple) {
  auto T = std::make_tuple(1, 3, 7);
  auto U = __orc_rt::apply_tuple(
      [](int A, int B, int C) { return std::make_tuple(A - B, B - C, C - A); },
      T);

  EXPECT_EQ(-2, std::get<0>(U));
  EXPECT_EQ(-4, std::get<1>(U));
  EXPECT_EQ(6, std::get<2>(U));

  auto V = __orc_rt::apply_tuple(
      [](int A, int B, int C) {
        return std::make_tuple(std::make_pair(A, char('A' + A)),
                               std::make_pair(B, char('A' + B)),
                               std::make_pair(C, char('A' + C)));
      },
      T);

  EXPECT_EQ(std::make_pair(1, 'B'), std::get<0>(V));
  EXPECT_EQ(std::make_pair(3, 'D'), std::get<1>(V));
  EXPECT_EQ(std::make_pair(7, 'H'), std::get<2>(V));
}

class apply_variadic {
  static int apply_one(int X) { return X + 1; }
  static char apply_one(char C) { return C + 1; }
  static std::string apply_one(std::string S) {
    return S.substr(0, S.size() - 1);
  }

public:
  template <typename... Ts> auto operator()(Ts &&... Items) {
    return std::make_tuple(apply_one(Items)...);
  }
};

TEST(STLExtrasTest, ApplyTupleVariadic) {
  auto Items = std::make_tuple(1, std::string("Test"), 'X');
  auto Values = __orc_rt::apply_tuple(apply_variadic(), Items);

  EXPECT_EQ(2, std::get<0>(Values));
  EXPECT_EQ("Tes", std::get<1>(Values));
  EXPECT_EQ('Y', std::get<2>(Values));
}
