//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class U> constexpr remove_cv_t<T> value_or(U&& v) const &;
// template<class U> constexpr remove_cv_t<T> value_or(U&& v) &&;

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"
#include "test_macros.h"

constexpr bool test() {
  // const &, has_value()
  {
    const std::expected<int, int> e(5);
    std::same_as<int> decltype(auto) x = e.value_or(10);
    assert(x == 5);
  }

  // const &, !has_value()
  {
    const std::expected<int, int> e(std::unexpect, 5);
    std::same_as<int> decltype(auto) x = e.value_or(10);
    assert(x == 10);
  }

  // &&, has_value()
  {
    std::expected<MoveOnly, int> e(std::in_place, 5);
    std::same_as<MoveOnly> decltype(auto) x = std::move(e).value_or(10);
    assert(x == 5);
  }

  // &&, !has_value()
  {
    std::expected<MoveOnly, int> e(std::unexpect, 5);
    std::same_as<MoveOnly> decltype(auto) x = std::move(e).value_or(10);
    assert(x == 10);
  }

  // LWG3424: return type is remove_cv_t<T>
  {
    const std::expected<const int, int> e(5);
    std::same_as<int> decltype(auto) x = e.value_or(10);
    assert(x == 5);
  }

  {
    std::expected<const int, int> e(std::unexpect, 5);
    std::same_as<int> decltype(auto) x = std::move(e).value_or(10);
    assert(x == 10);
  }

  // LWG3424: also check `volatile T` and `const volatile T`. Volatile reads
  // are not allowed in constant expressions, so we verify the return type
  // via decltype/declval without actually invoking value_or at runtime.
  {
    using E = std::expected<volatile int, int>;
    static_assert(std::is_same_v<decltype(std::declval<const E&>().value_or(0)), int>);
    static_assert(std::is_same_v<decltype(std::declval<E&&>().value_or(0)), int>);
  }
  {
    using E = std::expected<const volatile int, int>;
    static_assert(std::is_same_v<decltype(std::declval<const E&>().value_or(0)), int>);
    static_assert(std::is_same_v<decltype(std::declval<E&&>().value_or(0)), int>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
