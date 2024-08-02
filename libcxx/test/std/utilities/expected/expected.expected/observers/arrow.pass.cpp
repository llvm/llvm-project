//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr const T* operator->() const noexcept;
// constexpr T* operator->() noexcept;

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_macros.h"

// Test noexcept
template <class T>
concept ArrowNoexcept =
    requires(T t) {
      { t.operator->() } noexcept;
    };

static_assert(!ArrowNoexcept<int>);

static_assert(ArrowNoexcept<std::expected<int, int>>);
static_assert(ArrowNoexcept<const std::expected<int, int>>);

constexpr bool test() {
  // const
  {
    const std::expected<int, int> e(5);
    std::same_as<const int*> decltype(auto) x = e.operator->();
    assert(x == &(e.value()));
    assert(*x == 5);
  }

  // non-const
  {
    std::expected<int, int> e(5);
    std::same_as<int*> decltype(auto) x = e.operator->();
    assert(x == &(e.value()));
    assert(*x == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
