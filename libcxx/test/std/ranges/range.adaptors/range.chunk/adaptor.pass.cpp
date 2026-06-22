//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   std::views::chunk

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <ranges>
#include <utility>

constexpr bool test() {
  std::array<int, 8> array                       = {1, 2, 3, 4, 5, 6, 7, 8};
  std::ranges::ref_view<std::array<int, 8>> view = array | std::views::all;

  // Test `views::chunk(view, n)`
  {
    std::same_as<std::ranges::chunk_view<std::ranges::ref_view<std::array<int, 8>>>> decltype(auto) chunked =
        std::views::chunk(view, 2);
    assert(std::ranges::equal(*chunked.begin(), std::array{1, 2}));
    std::same_as<std::ranges::chunk_view<std::ranges::ref_view<std::array<int, 8>>>> decltype(auto) const_chunked =
        std::views::chunk(std::as_const(view), 2);
    assert(std::ranges::equal(*const_chunked.begin(), std::array{1, 2}));
  }

  // Test `views::chunk(n)(range)`
  {
    static_assert(noexcept(std::views::chunk(2)));
    /*__pipable*/ auto adaptor = std::views::chunk(3);
    std::same_as<std::ranges::chunk_view<std::ranges::ref_view<std::array<int, 8>>>> decltype(auto) chunked =
        adaptor(view);
    assert(std::ranges::equal(*chunked.begin(), std::array{1, 2, 3}));
    std::same_as<std::ranges::chunk_view<std::ranges::ref_view<std::array<int, 8>>>> decltype(auto) const_chunked =
        adaptor(std::as_const(view));
    assert(std::ranges::equal(*const_chunked.begin(), std::array{1, 2, 3}));
  }

  // Test `view | views::chunk`
  {
    std::same_as<std::ranges::chunk_view<std::ranges::ref_view<std::array<int, 8>>>> decltype(auto) chunked =
        view | std::views::chunk(4);
    assert(std::ranges::equal(*chunked.begin(), std::array{1, 2, 3, 4}));
    std::same_as<std::ranges::chunk_view<std::ranges::ref_view<std::array<int, 8>>>> decltype(auto) const_chunked =
        std::as_const(view) | std::views::chunk(4);
    assert(std::ranges::equal(*const_chunked.begin(), std::array{1, 2, 3, 4}));
  }

  // Test `views::chunk | adaptor`
  {
    /*__pipable*/ auto adaptors            = std::views::chunk(5) | std::views::join;
    std::ranges::input_range auto rejoined = view | adaptors;
    assert(std::ranges::equal(rejoined, view));
    std::ranges::input_range auto const_rejoined = std::as_const(view) | adaptors;
    assert(std::ranges::equal(const_rejoined, view));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
