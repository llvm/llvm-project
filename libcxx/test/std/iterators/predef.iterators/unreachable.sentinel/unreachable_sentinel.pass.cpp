//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// struct unreachable_sentinel_t;
// inline constexpr unreachable_sentinel_t unreachable_sentinel;

#include <iterator>

#include <cassert>
#include <concepts>
#include <type_traits>

#include "test_macros.h"

template<class T, class U>
concept weakly_equality_comparable_with = requires(const T& t, const U& u) {
  { t == u } -> std::same_as<bool>;
  { t != u } -> std::same_as<bool>;
  { u == t } -> std::same_as<bool>;
  { u != t } -> std::same_as<bool>;
};

constexpr bool test() {
  static_assert(std::is_empty_v<std::unreachable_sentinel_t>);
  static_assert(std::semiregular<std::unreachable_sentinel_t>);

  static_assert(std::same_as<decltype(std::unreachable_sentinel), const std::unreachable_sentinel_t>);

  auto sentinel = std::unreachable_sentinel;
  int i = 42;
  assert(i != sentinel);
  assert(sentinel != i);
  assert(!(i == sentinel));
  assert(!(sentinel == i));

  assert(&i != sentinel);
  assert(sentinel != &i);
  assert(!(&i == sentinel));
  assert(!(sentinel == &i));

  int *p = nullptr;
  assert(p != sentinel);
  assert(sentinel != p);
  assert(!(p == sentinel));
  assert(!(sentinel == p));

  static_assert( weakly_equality_comparable_with<std::unreachable_sentinel_t, int>);
  static_assert( weakly_equality_comparable_with<std::unreachable_sentinel_t, int*>);
  static_assert(!weakly_equality_comparable_with<std::unreachable_sentinel_t, void*>);
  ASSERT_NOEXCEPT(sentinel == p);
  ASSERT_NOEXCEPT(sentinel != p);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
