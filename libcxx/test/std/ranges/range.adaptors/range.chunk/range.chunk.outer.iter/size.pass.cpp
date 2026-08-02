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
//     constexpr auto outer_iterator::value_type::size() const
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;

#include <cassert>
#include <ranges>

#include "../types.h"

constexpr bool test() {
  int arr[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test `size()` when the range is fully divisible by the chunk size.
  {
    std::ranges::chunk_view<input_span<int>> chunked(input_span<int>(arr, 8), 4);
    auto outer = chunked.begin();

    assert((*outer).size() == 4);
    ++outer;
    assert((*outer).size() == 4);
  }

  // Test `size()` when the last chunk is smaller than the chunk size.
  {
    std::ranges::chunk_view<input_span<int>> chunked(input_span<int>(arr, 8), 3);
    auto outer = chunked.begin();

    assert((*outer).size() == 3);
    ++outer;
    assert((*outer).size() == 3);
    ++outer;
    assert((*outer).size() == 2);
  }

  // Test `size()` after partially consuming the current chunk via the inner iterator.
  {
    std::ranges::chunk_view<input_span<int>> chunked(input_span<int>(arr, 8), 3);
    auto outer = chunked.begin();
    auto inner = (*outer).begin();

    assert((*outer).size() == 3);
    ++inner;
    assert((*outer).size() == 2);
    ++inner;
    assert((*outer).size() == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
