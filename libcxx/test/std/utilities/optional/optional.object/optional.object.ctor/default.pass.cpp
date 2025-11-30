//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// constexpr optional() noexcept;

#include <cassert>
#include <optional>
#include <type_traits>

#include "test_macros.h"
#include "archetypes.h"

template <class T>
constexpr void test() {
  static_assert(std::is_nothrow_default_constructible_v<std::optional<T>>);
  static_assert(std::is_trivially_destructible_v<T> == std::is_trivially_destructible_v<std::optional<T>>);

  if constexpr (!std::is_lvalue_reference_v<T>) {
    static_assert(
        std::is_trivially_destructible_v<T> == std::is_trivially_destructible_v<typename std::optional<T>::value_type>);
  }

  {
    std::optional<T> opt;
    assert(static_cast<bool>(opt) == false);
  }
  {
    const std::optional<T> opt;
    assert(static_cast<bool>(opt) == false);
  }

  struct test_constexpr_ctor : public std::optional<T> {
    constexpr test_constexpr_ctor() {}
  };
}

TEST_CONSTEXPR_CXX23 void non_literal_test() {
  test<NonLiteralTypes::NoCtors>();
  test<NonLiteralTypes::NoCtors>();
}

constexpr bool test() {
  test<int>();
  test<int*>();
  test<ImplicitTypes::NoCtors>();
  test<NonTrivialTypes::NoCtors>();
  test<NonConstexprTypes::NoCtors>();
#if TEST_STD_VER >= 23
  non_literal_test();
#endif
#if TEST_STD_VER >= 26
  test<int&>();
  test<const int&>();
  test<int&>();
  test<NonLiteralTypes::NoCtors&>();
// TODO: optional<T&&> is not allowed.
#  if 0
  test<NonLiteralTypes::NoCtors&&>();
#  endif
#endif

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  {
    non_literal_test();
  }
  return 0;
}
