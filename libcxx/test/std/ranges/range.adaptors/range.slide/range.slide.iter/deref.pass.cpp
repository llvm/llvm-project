//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   constexpr value_type iterator::operator*() const;

#include <algorithm>
#include <cassert>
#include <compare>
#include <iterator>
#include <ranges>
#include <vector>

constexpr bool test() {
  std::vector<int> vector                                                 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::ranges::slide_view<std::ranges::ref_view<std::vector<int>>> slided = vector | std::views::slide(3);

  // Test `constexpr range_reference_v<V> inner_iterator::operator*() const`
  {
    std::same_as<std::span<int>> decltype(auto) v = *slided.begin();
    assert(std::ranges::equal(v, std::vector{1, 2, 3}));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
