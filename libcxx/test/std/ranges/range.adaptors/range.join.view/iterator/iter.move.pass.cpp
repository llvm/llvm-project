//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// friend constexpr decltype(auto) iter_move(const iterator& i);

#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  std::ranges::join_view jv(buffer);
  assert(std::ranges::iter_move(jv.begin()) == 1);
  ASSERT_SAME_TYPE(decltype(std::ranges::iter_move(jv.begin())), int&&);

  static_assert(noexcept(std::ranges::iter_move(std::declval<decltype(jv.begin())>())));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
