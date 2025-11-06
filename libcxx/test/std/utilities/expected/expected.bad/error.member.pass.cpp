//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// E& error() & noexcept;
// const E& error() const & noexcept;
// E&& error() && noexcept;
// const E&& error() const && noexcept;

#include <cassert>
#include <concepts>
#include <expected>
#include <utility>

template <class T>
concept ErrorNoexcept =
    requires(T&& t) {
      { std::forward<T>(t).error() } noexcept;
    };

static_assert(!ErrorNoexcept<int>);
static_assert(ErrorNoexcept<std::bad_expected_access<int>&>);
static_assert(ErrorNoexcept<std::bad_expected_access<int> const&>);
static_assert(ErrorNoexcept<std::bad_expected_access<int>&&>);
static_assert(ErrorNoexcept<std::bad_expected_access<int> const&&>);

void test() {
  // &
  {
    std::bad_expected_access<int> e(5);
    decltype(auto) i = e.error();
    static_assert(std::same_as<decltype(i), int&>);
    assert(i == 5);
  }

  // const &
  {
    const std::bad_expected_access<int> e(5);
    decltype(auto) i = e.error();
    static_assert(std::same_as<decltype(i), const int&>);
    assert(i == 5);
  }

  // &&
  {
    std::bad_expected_access<int> e(5);
    decltype(auto) i = std::move(e).error();
    static_assert(std::same_as<decltype(i), int&&>);
    assert(i == 5);
  }

  // const &&
  {
    const std::bad_expected_access<int> e(5);
    decltype(auto) i = std::move(e).error();
    static_assert(std::same_as<decltype(i), const int&&>);
    assert(i == 5);
  }
}

int main(int, char**) {
  test();
  return 0;
}
