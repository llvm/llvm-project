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
//     friend constexpr bool operator==(const inner_iterator& x, default_sentinel_t);

#include <cassert>
#include <iterator>
#include <ranges>
#include <vector>

#include "../types.h"

constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::ranges::chunk_view<input_span<int>> input_chunked = input_span<int>(vector) | std::views::chunk(3);

  // Test `friend constexpr bool operator==(const inner_iterator& x, default_sentinel_t)`
  {
    /*chunk_view::__inner_iterator*/ std::input_iterator auto it = (*input_chunked.begin()).begin();
    std::ranges::advance(it, 2);
    assert(it != std::default_sentinel);
    ++it;
    assert(it == std::default_sentinel);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
