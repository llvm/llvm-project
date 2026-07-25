//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// V models only input_range:
//   constexpr range_reference_v<V> inner_iterator::operator*() const;

#include <cassert>
#include <concepts>
#include <ranges>
#include <vector>

#include "../types.h"

constexpr bool test() {
  std::vector<int> vector                                = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::ranges::chunk_view<input_span<int>> input_chunked = input_span<int>(vector) | std::views::chunk(3);

  // Test `constexpr range_reference_v<V> inner_iterator::operator*() const`
  {
    /*chunk_view::__inner_iterator*/ std::input_iterator auto it = (*input_chunked.begin()).begin();
    std::same_as<int&> decltype(auto) v                          = *it;
    assert(v == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
