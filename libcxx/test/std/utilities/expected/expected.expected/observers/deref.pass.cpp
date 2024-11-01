//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr const T& operator*() const & noexcept;
// constexpr T& operator*() & noexcept;
// constexpr T&& operator*() && noexcept;
// constexpr const T&& operator*() const && noexcept;

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_macros.h"

// Test noexcept
template <class T>
concept DerefNoexcept =
    requires(T t) {
      { std::forward<T>(t).operator*() } noexcept;
    };

static_assert(!DerefNoexcept<int>);

static_assert(DerefNoexcept<std::expected<int, int>&>);
static_assert(DerefNoexcept<const std::expected<int, int>&>);
static_assert(DerefNoexcept<std::expected<int, int>&&>);
static_assert(DerefNoexcept<const std::expected<int, int>&&>);

constexpr bool test() {
  // non-const &
  {
    std::expected<int, int> e(5);
    decltype(auto) x = *e;
    static_assert(std::same_as<decltype(x), int&>);
    assert(&x == &(e.value()));
    assert(x == 5);
  }

  // const &
  {
    const std::expected<int, int> e(5);
    decltype(auto) x = *e;
    static_assert(std::same_as<decltype(x), const int&>);
    assert(&x == &(e.value()));
    assert(x == 5);
  }

  // non-const &&
  {
    std::expected<int, int> e(5);
    decltype(auto) x = *std::move(e);
    static_assert(std::same_as<decltype(x), int&&>);
    assert(&x == &(e.value()));
    assert(x == 5);
  }

  // const &&
  {
    const std::expected<int, int> e(5);
    decltype(auto) x = *std::move(e);
    static_assert(std::same_as<decltype(x), const int&&>);
    assert(&x == &(e.value()));
    assert(x == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
