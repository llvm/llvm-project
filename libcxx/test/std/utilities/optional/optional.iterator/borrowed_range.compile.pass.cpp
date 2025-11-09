//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

// template <class T> class optional<T&>::iterator;
// template <class T> class optional<T&>::const_iterator;
// template <class T>
// constexpr bool ranges::enable_borrowed_range<optional<T&>> = true;

#include <cassert>
#include <optional>
#include <ranges>

template <typename T>
constexpr void enable_borrowed_range() {
  static_assert(std::ranges::enable_borrowed_range<std::optional<T&>>);
}

template <typename T>
constexpr void borrowed_range() {
  static_assert(std::ranges::range<std::optional<T&>> == std::ranges::borrowed_range<std::optional<T&>>);
}

constexpr void test_enable_borrowed_range() {
  enable_borrowed_range<int>();
  enable_borrowed_range<const int>();
  enable_borrowed_range<int[]>();
  enable_borrowed_range<int[10]>();
  enable_borrowed_range<int()>();
}

constexpr void test_borrowed_range() {
  borrowed_range<int>();
  borrowed_range<const int>();
  borrowed_range<int[]>();
  borrowed_range<int[10]>();
  borrowed_range<int()>();
}

int main(int, char**) {
  test_enable_borrowed_range();
  test_borrowed_range();

  return 0;
}
