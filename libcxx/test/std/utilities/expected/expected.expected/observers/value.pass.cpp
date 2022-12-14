//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr const T& value() const &;
// constexpr T& value() &;
// constexpr T&& value() &&;
// constexpr const T&& value() const &&;

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"
#include "test_macros.h"

constexpr bool test() {
  // non-const &
  {
    std::expected<int, int> e(5);
    decltype(auto) x = e.value();
    static_assert(std::same_as<decltype(x), int&>);
    assert(&x == &(*e));
    assert(x == 5);
  }

  // const &
  {
    const std::expected<int, int> e(5);
    decltype(auto) x = e.value();
    static_assert(std::same_as<decltype(x), const int&>);
    assert(&x == &(*e));
    assert(x == 5);
  }

  // non-const &&
  {
    std::expected<int, int> e(5);
    decltype(auto) x = std::move(e).value();
    static_assert(std::same_as<decltype(x), int&&>);
    assert(&x == &(*e));
    assert(x == 5);
  }

  // const &&
  {
    const std::expected<int, int> e(5);
    decltype(auto) x = std::move(e).value();
    static_assert(std::same_as<decltype(x), const int&&>);
    assert(&x == &(*e));
    assert(x == 5);
  }

  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS

  // int
  {
    const std::expected<int, int> e(std::unexpect, 5);
    try {
      e.value();
      assert(false);
    } catch (const std::bad_expected_access<int>& ex) {
      assert(ex.error() == 5);
    }
  }

  // MoveOnly
  {
    std::expected<int, MoveOnly> e(std::unexpect, 5);
    try {
      std::move(e).value();
      assert(false);
    } catch (const std::bad_expected_access<MoveOnly>& ex) {
      assert(ex.error() == 5);
    }
  }

#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test();
  static_assert(test());
  testException();
  return 0;
}
