//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models only input_range
//     friend constexpr bool operator==(const outer_iterator& x, default_sentinel_t);
//     friend constexpr bool operator==(const inner_iterator& x, default_sentinel_t);

//   V models forward_range
//     friend constexpr bool operator==(const iterator& x, const iterator& y);
//     friend constexpr bool operator==(const iterator& x, default_sentinel_t);
//     friend constexpr bool operator<(const iterator& x, const iterator& y)
//       requires random_access_range<Base>;
//     friend constexpr bool operator>(const iterator& x, const iterator& y)
//       requires random_access_range<Base>;
//     friend constexpr bool operator<=(const iterator& x, const iterator& y)
//       requires random_access_range<Base>;
//     friend constexpr bool operator>=(const iterator& x, const iterator& y)
//       requires random_access_range<Base>;
//     friend constexpr auto operator<=>(const iterator& x, const iterator& y)
//       requires random_access_range<Base> &&
//                three_way_comparable<iterator_t<Base>>;

#include <cassert>
#include <compare>
#include <iterator>
#include <ranges>
#include <vector>

#include "test_range.h"
#include "../types.h"

constexpr bool test() {
  std::vector<int> vector                                                  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::ranges::chunk_view<std::ranges::ref_view<std::vector<int>>> chunked = vector | std::views::chunk(3);
  std::ranges::chunk_view<input_span<int>> input_chunked = input_span<int>(vector) | std::views::chunk(3);

  // Test `friend constexpr bool opreator==(const outer_iterator& x, default_sentinel_t)`
  {
    /*chunk_view::__outer_iterator*/ std::input_iterator auto it = input_chunked.begin();
    std::ranges::advance(it, 4);
    assert(it == std::default_sentinel);
  }

  // Test `friend constexpr bool operator==(const inner_iterator& x, default_sentinel_t)`
  {
    /*chunk_view::__inner_iterator*/ std::input_iterator auto it = (*input_chunked.begin()).begin();
    std::ranges::advance(it, 3);
    assert(it == std::default_sentinel);
  }

  // Test `friend constexpr bool operator==(const iterator& x, const iterator& y)`
  {
    assert(chunked.begin() == chunked.begin());
    assert(chunked.end() == chunked.end());
  }

  // Test `friend constexpr bool operator==(const iterator& x, default_sentinel)`
  {
    assert(chunked.end() == std::default_sentinel);
  }

  // Test `friend constexpr bool operator<(const iterator& x, const iterator& y)`
  {
    assert(chunked.begin() < chunked.end());
  }

  // Test `friend constexpr bool operator>(const iterator& x, const iterator& y)`
  {
    assert(chunked.end() > chunked.begin());
  }

  // Test `friend constexpr bool operator<=(const iterator& x, const iterator& y)`
  {
    assert(chunked.begin() <= chunked.begin());
    assert(chunked.begin() <= chunked.end());
  }

  // Test `friend constexpr bool operator>=(const iterator& x, const iterator& y)`
  {
    assert(chunked.end() >= chunked.end());
    assert(chunked.end() >= chunked.begin());
  }

  // Test `friend constexpr auto operator<=>(const iterator& x, const iterator& y)`
  {
    assert((chunked.begin() <=> chunked.begin()) == std::strong_ordering::equal);
    assert((chunked.begin() <=> chunked.end()) == std::strong_ordering::less);
    assert((chunked.end() <=> chunked.begin()) == std::strong_ordering::greater);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
