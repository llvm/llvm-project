//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class _Tp>
// inline constexpr empty_view<_Tp> empty{};

#include <ranges>
#include <cassert>

#include "test_macros.h"

template <class T>
constexpr void testType() {
  ASSERT_SAME_TYPE(decltype(std::views::empty<T>), const std::ranges::empty_view<T>);
  ASSERT_SAME_TYPE(decltype((std::views::empty<T>)), const std::ranges::empty_view<T>&);

  auto v = std::views::empty<T>;
  assert(std::ranges::empty(v));
}

struct Empty {};
struct BigType {
  char buff[8];
};

constexpr bool test() {

  testType<int>();
  testType<const int>();
  testType<int*>();
  testType<Empty>();
  testType<const Empty>();
  testType<BigType>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
