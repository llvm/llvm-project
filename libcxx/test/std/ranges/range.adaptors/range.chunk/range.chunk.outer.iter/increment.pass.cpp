//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models only input_range:
//     constexpr outer_iterator& operator++();
//     constexpr void operator++(int);

#include <algorithm>
#include <cassert>
#include <concepts>
#include <iterator>
#include <ranges>
#include <vector>

#include "../types.h"

constexpr bool test() {
  std::vector<int> vector                                = {1, 2, 3, 4, 5, 6, 7, 8};
  std::ranges::chunk_view<input_span<int>> input_chunked = input_span<int>(vector) | std::views::chunk(2);

  // Test `constexpr outer_iterator& operator++();`
  {
    /*chunk_view::__outer_iterator*/ std::input_iterator auto it = input_chunked.begin();
    assert(std::ranges::equal(*++it, std::vector{3, 4}));
  }

  // Test `constexpr void operator++(int);`
  {
    /*chunk_view::__outer_iterator*/ std::input_iterator auto it = input_chunked.begin();
    static_assert(std::same_as<decltype(it++), void>);
    it++;
    assert(std::ranges::equal(*it, std::vector{3, 4}));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
