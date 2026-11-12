//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr void operator*() const & noexcept;

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

static_assert(DerefNoexcept<std::expected<void, int>>);

constexpr bool test() {
  const std::expected<void, int> e;
  *e;
  static_assert(std::is_same_v<decltype(*e), void>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
