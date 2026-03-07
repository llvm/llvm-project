//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   constexpr V base() const& requires copy_constructible<V>;
//   constexpr V base() &&;

#include <array>
#include <cassert>
#include <concepts>
#include <memory>
#include <ranges>
#include <utility>

constexpr bool test() {
  std::array<int, 8> array                                                   = {1, 2, 3, 4, 5, 6, 7, 8};
  std::ranges::chunk_view<std::ranges::ref_view<std::array<int, 8>>> chunked = array | std::views::chunk(3);
  std::ranges::chunk_view<std::ranges::ref_view<const std::array<int, 8>>> const_chunked =
      std::as_const(array) | std::views::chunk(4);

  // Test `chunk_view.base()`
  {
    std::same_as<std::array<int, 8>&> decltype(auto) base = chunked.base().base();
    assert(std::addressof(base) == std::addressof(array));

    std::same_as<const std::array<int, 8>&> decltype(auto) const_base = const_chunked.base().base();
    assert(std::addressof(const_base) == std::addressof(array));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
