//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   friend constexpr bool operator==(const iterator& x, const iterator& y)
//   friend constexpr bool operator<(const iterator& x, const iterator& y)
//     requires random_access_range<Base>;
//   friend constexpr bool operator>(const iterator& x, const iterator& y)
//     requires random_access_range<Base>;
//   friend constexpr bool operator<=(const iterator& x, const iterator& y)
//     requires random_access_range<Base>;
//   friend constexpr bool operator>=(const iterator& x, const iterator& y)
//     requires random_access_range<Base>;
//   friend constexpr auto operator<=>(const iterator& x, const iterator& y)
//     requires random_access_range<Base> &&
//              three_way_comparable<iterator_t<Base>>;

#include <algorithm>
#include <cassert>
#include <compare>
#include <iterator>
#include <ranges>
#include <vector>

constexpr bool test() {
  std::vector<int> vector                                                 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::ranges::slide_view<std::ranges::ref_view<std::vector<int>>> slided = vector | std::views::slide(3);

  // Test `friend constexpr bool operator==(const iterator& x, const iterator& y)`
  {
    assert(slided.begin() == slided.begin());
    assert(slided.end() == slided.end());
  }

  // Test `friend constexpr bool operator<(const iterator& x, const iterator& y)`
  {
    assert(slided.begin() < slided.end());
  }

  // Test `friend constexpr bool operator>(const iterator& x, const iterator& y)`
  {
    assert(slided.end() > slided.begin());
  }

  // Test `friend constexpr bool operator>=(const iterator& x, const iterator& y)`
  {
    assert(slided.begin() <= slided.begin());
    assert(slided.begin() <= slided.end());
  }

  // Test `friend constexpr bool operator>=(const iterator& x, const iterator& y)`
  {
    assert(slided.end() >= slided.end());
    assert(slided.end() >= slided.begin());
  }

  // Test `friend constexpr auto operator<=>(const iterator& x, const iterator& y)`
  {
    assert((slided.begin() <=> slided.begin()) == std::strong_ordering::equal);
    assert((slided.begin() <=> slided.end()) == std::strong_ordering::less);
    assert((slided.end() <=> slided.begin()) == std::strong_ordering::greater);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
