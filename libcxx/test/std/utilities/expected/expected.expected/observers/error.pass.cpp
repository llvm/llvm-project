//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr const E& error() const & noexcept;
// constexpr E& error() & noexcept;
// constexpr E&& error() && noexcept;
// constexpr const E&& error() const && noexcept;

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_macros.h"

// Test noexcept
template <class T>
concept ErrorNoexcept =
    requires(T t) {
      { std::forward<T>(t).error() } noexcept;
    };

static_assert(!ErrorNoexcept<int>);

static_assert(ErrorNoexcept<std::expected<int, int>&>);
static_assert(ErrorNoexcept<const std::expected<int, int>&>);
static_assert(ErrorNoexcept<std::expected<int, int>&&>);
static_assert(ErrorNoexcept<const std::expected<int, int>&&>);

constexpr bool test() {
  // non-const &
  {
    std::expected<int, int> e(std::unexpect, 5);
    decltype(auto) x = e.error();
    static_assert(std::same_as<decltype(x), int&>);
    assert(x == 5);
  }

  // const &
  {
    const std::expected<int, int> e(std::unexpect, 5);
    decltype(auto) x = e.error();
    static_assert(std::same_as<decltype(x), const int&>);
    assert(x == 5);
  }

  // non-const &&
  {
    std::expected<int, int> e(std::unexpect, 5);
    decltype(auto) x = std::move(e).error();
    static_assert(std::same_as<decltype(x), int&&>);
    assert(x == 5);
  }

  // const &&
  {
    const std::expected<int, int> e(std::unexpect, 5);
    decltype(auto) x = std::move(e).error();
    static_assert(std::same_as<decltype(x), const int&&>);
    assert(x == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
