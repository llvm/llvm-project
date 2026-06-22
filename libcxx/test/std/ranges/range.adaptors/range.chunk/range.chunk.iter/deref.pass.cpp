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
//     constexpr value_type outer_iterator::operator*() const;
//     constexpr inner_iterator outer_iterator::value_type::begin() const noexcept;
//     constexpr default_sentinel_t outer_iterator::value_type::end() const noexcept;
//     constexpr range_reference_v<V> inner_iterator::operator*() const;

//   V models forward_range:
//     constexpr value_type iterator::operator*() const;

#include <algorithm>
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

  // Test `constexpr value_type outer_iterator::operator*() const`
  {
    static_assert(std::ranges::input_range<decltype(*input_chunked.begin())>);
  }

  // Test `constexpr inner_iterator outer_iterator::value_type::begin() const noexcept`
  {
    /*chunk_view::__outer_iterator::value_type*/ std::ranges::input_range auto inner = *input_chunked.begin();
    assert(*inner.begin() == *vector.begin());
    static_assert(noexcept(inner.begin()));
  }

  // Test `constexpr default_sentinel_t outer_iterator::value_type::end() const noexcept`
  {
    /*chunk_view::__outer_iterator::value_type*/ std::ranges::input_range auto inner = *input_chunked.begin();
    [[maybe_unused]] std::same_as<std::default_sentinel_t> auto it                   = inner.end();
    static_assert(noexcept((inner.end())));
  }

  // Test `constexpr value_type iterator::operator*() const`
  {
    /*chunk_view::__inner_iterator*/ std::input_iterator auto it = (*input_chunked.begin()).begin();
    std::same_as<int> decltype(auto) v                           = *it;
    assert(v == 1);
  }

  // Test `constexpr range_reference_v<V> inner_iterator::operator*() const`
  {
    std::same_as<int&> decltype(auto) v = *(*chunked.begin()).begin();
    assert(v == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
