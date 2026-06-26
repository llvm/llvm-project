//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   constexpr explicit chunk_view(_View __base, range_difference_t<_View> __n);

#include <algorithm>
#include <cassert>
#include <ranges>
#include <utility>
#include <vector>

#include "test_convertible.h"
#include "test_range.h"
#include "types.h"

constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test `chunk_view(_View, range_difference_t<_View>)` when V models only `input_range`
  {
    static_assert(!test_convertible<std::ranges::chunk_view<input_span<int>>, input_span<int>, std::ptrdiff_t>());

    std::ranges::chunk_view<input_span<int>> chunked(input_span(vector.data(), 8), 3);
    assert(std::ranges::equal(*chunked.begin(), std::vector{1, 2, 3}));
  }

  // Test `chunk_view(_View, range_difference_t<_View>)` when V models `forward_range`
  {
    static_assert(!test_convertible<std::ranges::chunk_view<input_span<int>>,
                                    std::ranges::ref_view<std::vector<int>>,
                                    std::ptrdiff_t>());

    std::ranges::chunk_view<input_span<int>> chunked(std::ranges::ref_view(vector), 3);
    assert(std::ranges::equal(*chunked.begin(), std::vector{1, 2, 3}));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
