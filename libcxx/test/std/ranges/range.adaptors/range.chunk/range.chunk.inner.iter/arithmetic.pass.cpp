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
//     friend constexpr difference_type operator-(default_sentinel_t y, const inner_iterator& x)
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;
//     friend constexpr difference_type operator-(const inner_iterator& x, default_sentinel_t y)
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;

#include <cassert>
#include <concepts>
#include <iterator>
#include <ranges>
#include <vector>

#include "../types.h"

constexpr bool test() {
  std::vector<int> vector                                = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::ranges::chunk_view<input_span<int>> input_chunked = input_span<int>(vector) | std::views::chunk(3);

  // Test `friend constexpr difference_type operator-(default_sentinel_t, const inner_iterator& x)`
  {
    using InnerIterator = std::ranges::iterator_t<std::ranges::range_reference_t<decltype(input_chunked)>>;
    static_assert(std::same_as<decltype(std::default_sentinel - (*input_chunked.begin()).begin()),
                               InnerIterator::difference_type>);
    assert(std::default_sentinel - (*input_chunked.begin()).begin() == 3);
  }

  // Test `friend constexpr difference_type operator-(const inner_iterator& x, default_sentinel_t)`
  {
    assert((*input_chunked.begin()).begin() - std::default_sentinel == -3);
  }

  // Test `operator-(default_sentinel_t, inner_iterator)` when the chunk itself is smaller than
  // the chunk size (`__remainder_` stays larger than `__dist`, so `__dist` is the returned minimum).
  {
    std::vector<int> uneven_vector                          = {1, 2, 3, 4, 5, 6, 7, 8};
    std::ranges::chunk_view<input_span<int>> uneven_chunked = input_span<int>(uneven_vector) | std::views::chunk(3);
    auto outer                                              = uneven_chunked.begin();
    ++outer;
    ++outer;
    assert(std::default_sentinel - (*outer).begin() == 2);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
