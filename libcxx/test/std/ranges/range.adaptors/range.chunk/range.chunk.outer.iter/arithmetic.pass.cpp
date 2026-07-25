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
//     friend constexpr difference_type operator-(default_sentinel_t t, const outer_iterator& i)
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;
//     friend constexpr difference_type operator-(const outer_iterator& i, default_sentinel_t t)
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;

#include <cassert>
#include <iterator>
#include <ranges>
#include <vector>

#include "../types.h"

constexpr bool test() {
  std::vector<int> vector                                = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::ranges::chunk_view<input_span<int>> input_chunked = input_span<int>(vector) | std::views::chunk(3);

  // Test `friend constexpr difference_type operator-(default_sentinel_t t, const outer_iterator& i)`
  {
    assert(std::default_sentinel - input_chunked.begin() == 4);
  }

  // Test `friend constexpr difference_type operator-(const outer_iterator& i, default_sentinel_t)`
  {
    assert(input_chunked.begin() - std::default_sentinel == -4);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
