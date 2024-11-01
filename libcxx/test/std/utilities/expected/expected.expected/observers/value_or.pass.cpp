//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class U> constexpr T value_or(U&& v) const &;
// template<class U> constexpr T value_or(U&& v) &&;

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
