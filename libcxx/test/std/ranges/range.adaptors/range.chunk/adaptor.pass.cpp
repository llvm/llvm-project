//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::chunk

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <span>

#include "test_range.h"
#include "types.h"

constexpr bool test() {
  std::array array    = {1, 1, 1, 2, 2, 2, 3, 3};
  std::span<int> span = {array.data(), 8};

  // Test `views::chunk(view, n)`
  {
    auto chunked = std::views::chunk(span, 2);
    static_assert(std::same_as<decltype(chunked), std::ranges::chunk_view<std::span<int>>>);
    assert(std::ranges::equal(*chunked.begin(), std::array{1, 1}));
  }

  // Test `views::chunk(input_view, n)`
  {
    auto input   = exactly_input_view<decltype(span)>(span);
    auto chunked = std::views::chunk(input, 3);
    assert(std::ranges::equal(*chunked.begin(), std::array{1, 1, 1}));
  }

  // Test `views::chunk(n)(range)`
  {
    auto adaptor = std::views::chunk(4);
    auto chunked = adaptor(span);
    static_assert(std::same_as<decltype(chunked), std::ranges::chunk_view<std::span<int>>>);
    assert(std::ranges::equal(*chunked.begin(), std::array{1, 1, 1, 2}));
  }

  // Test `view | views::chunk`
  {
    auto chunked = span | std::views::chunk(5);
    static_assert(std::same_as<decltype(chunked), std::ranges::chunk_view<std::span<int>>>);
    static_assert(std::ranges::random_access_range<decltype(chunked)>);
    assert(std::ranges::equal(*chunked.begin(), std::array{1, 1, 1, 2, 2}));
  }

  // Test `views::chunk | adaptor`
  {
    auto multi_adaptor = std::views::chunk(1) | std::views::join;
    auto rejoined      = span | multi_adaptor;
    assert(std::ranges::equal(rejoined, span));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}