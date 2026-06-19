//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23, c++26

// constexpr bool has_error() const noexcept;

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "../../types.h"

// Test noexcept
template <class T>
concept HasErrorNoexcept =
    requires(T t) {
      { t.has_error() } noexcept;
    };

struct Foo {};
static_assert(!HasErrorNoexcept<Foo>);

static_assert(HasErrorNoexcept<std::expected<int, int>>);
static_assert(HasErrorNoexcept<const std::expected<int, int>>);

constexpr bool test() {
  // has_error
  {
    const std::expected<int, int> e(std::unexpect, 5);
    assert(e.has_error());
  }

  // !has_error
  {
    const std::expected<int, int> e(5);
    assert(!e.has_error());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
