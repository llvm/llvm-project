//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::ranges::elements_of;

#include <ranges>

#include <concepts>
#include <memory>
#include <vector>

constexpr bool test() {
  {
    auto elements_of = std::ranges::elements_of(std::vector<int>());
    static_assert(
        std::same_as<decltype(elements_of), std::ranges::elements_of<std::vector<int>&&, std::allocator<std::byte>>>);
    static_assert(std::same_as<decltype(elements_of.range), std::vector<int>&&>);
    static_assert(std::same_as<decltype(elements_of.allocator), std::allocator<std::byte>>);
  }
  {
    auto elements_of = std::ranges::elements_of(std::vector<int>(), std::allocator<int>());
    static_assert(
        std::same_as<decltype(elements_of), std::ranges::elements_of<std::vector<int>&&, std::allocator<int>>>);
    static_assert(std::same_as<decltype(elements_of.range), std::vector<int>&&>);
    static_assert(std::same_as<decltype(elements_of.allocator), std::allocator<int>>);
  }
  return true;
}

int main() {
  test();
  static_assert(test());
}
