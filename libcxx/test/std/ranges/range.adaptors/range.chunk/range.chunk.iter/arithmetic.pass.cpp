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
//     friend constexpr difference_type operator-(default_sentinel_t t, const outer_iterator& i)
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;
//     friend constexpr difference_type operator-(const outer_iterator& i, default_sentinel_t t)
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;
//     friend constexpr difference_type operator-(default_sentinel_t y, const inner_iterator& x)
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;
//     friend constexpr difference_type operator-(const inner_iterator& x, default_sentinel_t y)
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;

//   V models forward_range
//     friend constexpr iterator operator+(const iterator& i, difference_type n)
//       requires random_access_range<Base>;
//     friend constexpr iterator operator+(difference_type n, const iterator& i)
//       requires random_access_range<Base>;
//     friend constexpr iterator operator-(const iterator& i, difference_type n)
//       requires random_access_range<Base>;
//     friend constexpr difference_type operator-(const iterator& x, const iterator& y)
//       requires sized_sentinel_for<iterator_t<Base>, iterator_t<Base>>;
//     friend constexpr difference_type operator-(default_sentinel_t y, const iterator& x)
//       requires sized_sentinel_for<sentinel_t<Base>, iterator_t<Base>>;
//     friend constexpr difference_type operator-(const iterator& x, default_sentinel_t y)
//       requires sized_sentinel_for<sentinel_t<Base>, iterator_t<Base>>;

#include <cassert>
#include <iterator>
#include <ranges>
#include <vector>

#include "test_range.h"
#include "../types.h"

constexpr bool test() {
  std::vector<int> vector                                                  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::ranges::chunk_view<std::ranges::ref_view<std::vector<int>>> chunked = vector | std::views::chunk(3);
  std::ranges::chunk_view<input_span<int>> input_chunked = input_span<int>(vector) | std::views::chunk(3);

  // Test `friend constexpr difference_type operator-(default_sentinel_t t, const outer_iterator& i)`
  {
    assert(std::default_sentinel - input_chunked.begin() == 4);
  }

  // Test `friend constexpr difference_type operator-(const outer_iterator& i, default_sentinel_t)`
  {
    assert(input_chunked.begin() - std::default_sentinel == -4);
  }

  // Test `friend constexpr difference_type operator-(default_sentinel_t, const inner_iterator& x)`
  {
    assert(std::default_sentinel - (*input_chunked.begin()).begin() == 3);
  }

  // Test `friend constexpr difference_type operator-(const inner_iterator& x, default_sentinel_t)`
  {
    assert((*input_chunked.begin()).begin() - std::default_sentinel == -3);
  }

  // Test `friend constexpr iterator operator+(const iterator& i, difference_type n)`
  {
    assert(chunked.begin() + 4 == chunked.end());
  }

  // Test `friend constexpr iterator operator+(difference_type n, const iterator& i)`
  {
    assert(4 + chunked.begin() == chunked.end());
  }

  // Test `friend constexpr iterator operator-(const iterator& i, difference_type n)`
  {
    assert(chunked.end() - 4 == chunked.begin());
  }

  // Test `friend constexpr difference_type operator-(const iterator& x, const iterator& y)`
  {
    assert(chunked.end() - chunked.begin() == 4);
  }

  // Test `friend constexpr difference_type operator-(default_sentinel_t y, const iterator& x)`
  {
    assert(std::default_sentinel - chunked.begin() == 4);
  }

  // Test `friend constexpr difference_type operator-(const iterator& x, default_sentinel_t y)`
  {
    assert(chunked.begin() - std::default_sentinel == -4);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
