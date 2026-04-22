//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// class enumerate_view::iterator

// friend constexpr bool operator==(const iterator& x, const iterator& y) noexcept;

#include <cstddef>
#include <cassert>
#include <ranges>

#include "test_iterators.h"

#include "../types.h"

constexpr bool test() {
  constexpr std::size_t arrSize = 3uz;
  int buff[arrSize]             = {94, 82, 49};

  using EnumerateView = std::ranges::enumerate_view<RangeView>;
  const RangeView range(buff, buff + arrSize);

  std::same_as<EnumerateView> decltype(auto) ev = std::views::enumerate(range);

  auto it1 = ev.begin();
  auto it2 = it1 + arrSize; // End of the array.

  assert(it1 == it1);
  assert(it1 != it2);
  assert(it2 != it1);
  assert(it2 == ev.end());
  assert(ev.end() == it2);

  // Increment x3 to the end of the array.
  ++it1;
  ++it1;
  ++it1;

  assert(it1 == ev.end());
  assert(ev.end() == it1);

  static_assert(noexcept(it1 == it1));
  static_assert(noexcept(it1 != it2));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
