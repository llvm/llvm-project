//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   constexpr auto size() requires sized_range<_View>;
//   constexpr auto size() const requires sized_range<const _View>;

#include <cassert>
#include <ranges>

#include "test_range.h"
#include "types.h"

constexpr bool test() {
  int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Test `chunk_view.size()` when V models only `input_range`
  {
    static_assert(std::ranges::sized_range<std::ranges::chunk_view<input_span<int>>>);
    static_assert(!std::ranges::sized_range<const std::ranges::chunk_view<input_span<int>>>);

    auto chunked = input_span<int>(arr, 12) | std::views::chunk(3);
    assert(chunked.size() == 4);
  }

  // Test `chunk_view.size()` when V models `forward_range`
  {
    static_assert(std::ranges::sized_range<std::ranges::chunk_view<std::ranges::ref_view<int[12]>>>);
    static_assert(std::ranges::sized_range<const std::ranges::chunk_view<std::ranges::ref_view<int[12]>>>);

    auto chunked = std::ranges::ref_view(arr) | std::views::chunk(3);
    assert(chunked.size() == 4);
    const auto& const_chunked = chunked;
    assert(const_chunked.size() == 4);
  }

  // Test `chunk_view.size()` when the range is not fully divisible
  {
    auto chunked = std::ranges::ref_view(arr) | std::views::chunk(5);
    assert(chunked.size() == 3);
  }

  // Test `chunk_view.size()` when the range is empty
  {
    static_assert(std::ranges::sized_range<std::ranges::empty_view<int>>);
    auto chunked = std::views::empty<int> | std::views::chunk(3);
    assert(chunked.size() == 0);
  }

  // Test `chunk_view.size()` when chunk size is larger than the range
  {
    auto chunked = std::ranges::ref_view(arr) | std::views::chunk(100);
    assert(chunked.size() == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
