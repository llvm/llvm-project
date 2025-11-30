//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// constexpr optional(nullopt_t) noexcept;

#include <cassert>
#include <optional>
#include <type_traits>

#include "archetypes.h"

#include "test_macros.h"

template <class T>
constexpr void test() {
  static_assert(std::is_nothrow_constructible_v<std::optional<T>, std::nullopt_t&>);
  static_assert(std::is_trivially_destructible_v<T> == std::is_trivially_destructible_v<std::optional<T>>);
  static_assert(
      std::is_trivially_destructible_v<T> == std::is_trivially_destructible_v<typename std::optional<T>::value_type>);

  constexpr std::optional<T> opt(std::nullopt);
  assert(!static_cast<bool>(opt));

  struct test_constexpr_ctor : public std::optional<T> {
    constexpr test_constexpr_ctor() {}
  };
}

template <class T>
TEST_CONSTEXPR_CXX23 void rt_test() {
  static_assert(std::is_nothrow_constructible_v<std::optional<T>, std::nullopt_t&>);
  static_assert(std::is_trivially_destructible_v<T> == std::is_trivially_destructible_v<std::optional<T>>);
  static_assert(
      std::is_trivially_destructible_v<T> == std::is_trivially_destructible_v<typename std::optional<T>::value_type>);

  const std::optional<T> opt(std::nullopt);
  assert(!static_cast<bool>(opt));
}

TEST_CONSTEXPR_CXX23 bool test_non_literal() {
  rt_test<NonLiteralTypes::NoCtors>();
  return true;
}

constexpr bool test() {
  test<int>();
  test<int*>();
  test<ImplicitTypes::NoCtors>();
  test<NonTrivialTypes::NoCtors>();
  test<NonConstexprTypes::NoCtors>();
#if TEST_STD_VER >= 23
  test_non_literal();
#endif
  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  {
    test_non_literal();
  }

  return 0;
}
