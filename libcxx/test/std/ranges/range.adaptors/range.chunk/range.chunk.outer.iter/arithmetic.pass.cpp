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
#include <concepts>
#include <iterator>
#include <ranges>
#include <vector>

#include "../types.h"

constexpr bool test() {
  std::vector<int> vector                                = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::ranges::chunk_view<input_span<int>> input_chunked = input_span<int>(vector) | std::views::chunk(3);

  // Test `friend constexpr difference_type operator-(default_sentinel_t t, const outer_iterator& i)`
  {
    using OuterIterator = std::ranges::iterator_t<decltype(input_chunked)>;
    static_assert(
        std::same_as<decltype(std::default_sentinel - input_chunked.begin()), OuterIterator::difference_type>);
    assert(std::default_sentinel - input_chunked.begin() == 4);
  }

  // Test `friend constexpr difference_type operator-(const outer_iterator& i, default_sentinel_t)`
  {
    assert(input_chunked.begin() - std::default_sentinel == -4);
  }

  // Test `operator-(default_sentinel_t, outer_iterator)` when the last chunk is smaller than the chunk size
  // (the outer iterator sits at the start of a partial chunk, so `__dist < __remainder_`).
  {
    std::vector<int> uneven_vector                                = {1, 2, 3, 4, 5, 6, 7, 8};
    std::ranges::chunk_view<input_span<int>> uneven_chunked        = input_span<int>(uneven_vector) | std::views::chunk(3);
    auto it = uneven_chunked.begin();
    ++it;
    ++it;
    assert(std::default_sentinel - it == 1);
    assert(it - std::default_sentinel == -1);

    // Fully consume the range: the outer iterator now equals `default_sentinel`.
    ++it;
    assert(it == uneven_chunked.end());
    assert(std::default_sentinel - it == 0);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
