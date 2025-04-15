//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr explicit operator bool() const noexcept;

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_macros.h"

// Test noexcept
template <class T>
concept OpBoolNoexcept =
    requires(T t) {
      { static_cast<bool>(t) } noexcept;
    };

struct Foo {};
static_assert(!OpBoolNoexcept<Foo>);

static_assert(OpBoolNoexcept<std::expected<void, int>>);
static_assert(OpBoolNoexcept<const std::expected<void, int>>);

// Test explicit
static_assert(!std::is_convertible_v<std::expected<void, int>, bool>);

constexpr bool test() {
  // has_value
  {
    const std::expected<void, int> e;
    assert(static_cast<bool>(e));
  }

  // !has_value
  {
    const std::expected<void, int> e(std::unexpect, 5);
    assert(!static_cast<bool>(e));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
