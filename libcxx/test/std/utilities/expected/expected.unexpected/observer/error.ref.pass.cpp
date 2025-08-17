//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr E& error() & noexcept;

#include <cassert>
#include <concepts>
#include <expected>
#include <utility>

template <class T>
concept ErrorNoexcept =
    requires(T& t) {
      { t.error() } noexcept;
    };

static_assert(!ErrorNoexcept<int>);
static_assert(ErrorNoexcept<std::unexpected<int>>);

constexpr bool test() {
  std::unexpected<int> unex(5);
  decltype(auto) i = unex.error();
  static_assert(std::same_as<decltype(i), int&>);
  assert(i == 5);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
