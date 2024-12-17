//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class... Args>
//   constexpr explicit expected(unexpect_t, Args&&... args);
//
// Constraints: is_constructible_v<E, Args...> is true.
//
// Effects: Direct-non-list-initializes unex with std::forward<Args>(args)....
//
// Postconditions: has_value() is false.
//
// Throws: Any exception thrown by the initialization of unex.

#include <cassert>
#include <expected>
#include <tuple>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"
#include "test_macros.h"
#include "../../types.h"

// Test Constraints:
static_assert(std::is_constructible_v<std::expected<int, int>, std::unexpect_t>);
static_assert(std::is_constructible_v<std::expected<int, int>, std::unexpect_t, int>);

// !is_constructible_v<T, Args...>
struct foo {};
static_assert(!std::is_constructible_v<std::expected<int, foo>, std::unexpect_t, int>);

// test explicit
template <class T>
void conversion_test(T);

template <class T, class... Args>
concept ImplicitlyConstructible = requires(Args&&... args) { conversion_test<T>({std::forward<Args>(args)...}); };
static_assert(ImplicitlyConstructible<int, int>);

static_assert(!ImplicitlyConstructible<std::expected<int, int>, std::unexpect_t>);
static_assert(!ImplicitlyConstructible<std::expected<int, int>, std::unexpect_t, int>);

struct CopyOnly {
  int i;
  constexpr CopyOnly(int ii) : i(ii) {}
  CopyOnly(const CopyOnly&) = default;
  CopyOnly(CopyOnly&&)      = delete;
  friend constexpr bool operator==(const CopyOnly& mi, int ii) { return mi.i == ii; }
};

template <class T, class V = int>
constexpr void testInt() {
  std::expected<V, T> e(std::unexpect, 5);
  assert(!e.has_value());
  assert(e.error() == 5);
}

template <class T, class V = int>
constexpr void testLValue() {
  T t(5);
  std::expected<V, T> e(std::unexpect, t);
  assert(!e.has_value());
  assert(e.error() == 5);
}

template <class T, class V = int>
constexpr void testRValue() {
  std::expected<V, T> e(std::unexpect, T(5));
  assert(!e.has_value());
  assert(e.error() == 5);
}

constexpr bool test() {
  testInt<int>();
  testInt<CopyOnly>();
  testInt<MoveOnly>();
  testInt<TailClobberer<1>, bool>();
  testLValue<int>();
  testLValue<CopyOnly>();
  testLValue<TailClobberer<1>, bool>();
  testRValue<int>();
  testRValue<MoveOnly>();
  testRValue<TailClobberer<1>, bool>();

  // no arg
  {
    std::expected<int, int> e(std::unexpect);
    assert(!e.has_value());
    assert(e.error() == 0);
  }

  // one arg
  {
    std::expected<int, int> e(std::unexpect, 5);
    assert(!e.has_value());
    assert(e.error() == 5);
  }

  // multi args
  {
    std::expected<int, std::tuple<int, short, MoveOnly>> e(std::unexpect, 1, short{2}, MoveOnly(3));
    assert(!e.has_value());
    assert((e.error() == std::tuple<int, short, MoveOnly>(1, short{2}, MoveOnly(3))));
  }

  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Throwing {
    Throwing(int) { throw Except{}; };
  };

  try {
    std::expected<int, Throwing> u(std::unexpect, 5);
    assert(false);
  } catch (Except) {
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test();
  static_assert(test());
  testException();
  return 0;
}
