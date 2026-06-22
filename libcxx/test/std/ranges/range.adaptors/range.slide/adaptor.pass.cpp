//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   std::views::slide

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <ranges>
#include <utility>

#include "test_range.h"

constexpr bool test() {
  std::array<int, 8> array                       = {1, 2, 3, 4, 5, 6, 7, 8};
  std::ranges::ref_view<std::array<int, 8>> view = array | std::views::all;

  // Test `views::slide(view, n)`
  {
    std::same_as<std::ranges::slide_view<std::ranges::ref_view<std::array<int, 8>>>> decltype(auto) slided =
        std::views::slide(view, 2);
    assert(std::ranges::equal(*slided.begin(), std::array{1, 2}));
    std::same_as<std::ranges::slide_view<std::ranges::ref_view<std::array<int, 8>>>> decltype(auto) const_slided =
        std::views::slide(std::as_const(view), 2);
    assert(std::ranges::equal(*const_slided.begin(), std::array{1, 2}));
  }

  // Test `views::slide(n)(range)`
  {
    static_assert(noexcept(std::views::slide(3)));
    /*__pipable*/ auto adaptor = std::views::slide(3);
    std::same_as<std::ranges::slide_view<std::ranges::ref_view<std::array<int, 8>>>> decltype(auto) slided =
        adaptor(view);
    assert(std::ranges::equal(*slided.begin(), std::array{1, 2, 3}));
    std::same_as<std::ranges::slide_view<std::ranges::ref_view<std::array<int, 8>>>> decltype(auto) const_slided =
        adaptor(std::as_const(view));
    assert(std::ranges::equal(*const_slided.begin(), std::array{1, 2, 3}));
  }

  // Test `view | views::slide`
  {
    std::same_as<std::ranges::slide_view<std::ranges::ref_view<std::array<int, 8>>>> decltype(auto) slided =
        view | std::views::slide(4);
    assert(std::ranges::equal(*slided.begin(), std::array{1, 2, 3, 4}));
    std::same_as<std::ranges::slide_view<std::ranges::ref_view<std::array<int, 8>>>> decltype(auto) const_slided =
        std::as_const(view) | std::views::slide(4);
    assert(std::ranges::equal(*const_slided.begin(), std::array{1, 2, 3, 4}));
  }

  // Test `views::slide | adaptor`
  {
    /*__pipable*/ auto adaptors            = std::views::slide(3) | std::views::join;
    std::ranges::input_range auto rejoined = view | adaptors;
    assert(std::ranges::equal(
        rejoined, std::array{1, 2, 3, /*|*/ 2, 3, 4, /*|*/ 3, 4, 5, /*|*/ 4, 5, 6, /*|*/ 5, 6, 7, /*|*/ 6, 7, 8}));
    std::ranges::input_range auto const_rejoined = std::as_const(view) | adaptors;
    assert(std::ranges::equal(
        const_rejoined,
        std::array{1, 2, 3, /*|*/ 2, 3, 4, /*|*/ 3, 4, 5, /*|*/ 4, 5, 6, /*|*/ 5, 6, 7, /*|*/ 6, 7, 8}));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
