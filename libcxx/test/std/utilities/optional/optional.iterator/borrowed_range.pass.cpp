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
constexpr bool enable_borrowed_range() {
  {
    assert(std::ranges::enable_borrowed_range<std::optional<T&>>);
  }
  return true;
}

template <typename T>
constexpr bool borrowed_range() {
  if (std::ranges::range<std::optional<T&>>) {
    assert(std::ranges::borrowed_range<std::optional<T&>>);
  } else {
    assert(!std::ranges::borrowed_range<std::optional<T&>>);
    return false;
  }

  return true;
}

constexpr bool test_enable_borrowed_range() {
  assert(enable_borrowed_range<int>());
  assert(enable_borrowed_range<const int>());
  assert(enable_borrowed_range<int[]>());
  assert(enable_borrowed_range<int[10]>());
  assert(enable_borrowed_range<int()>());

  return true;
}

constexpr bool test_borrowed_range() {
  assert(borrowed_range<int>());
  assert(borrowed_range<const int>());
  assert(!borrowed_range<int[]>());
  assert(!borrowed_range<int[10]>());
  assert(!borrowed_range<int()>());

  return true;
}

int main(int, char**) {
  {
    static_assert(test_enable_borrowed_range());
    assert(test_enable_borrowed_range());
  }

  {
    static_assert(test_borrowed_range());
    assert(test_enable_borrowed_range());
  }

  return 0;
}