//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class G>
//   constexpr explicit(!is_convertible_v<G, E>) expected(unexpected<G>&& e);
//
// Let GF be G
//
// Constraints: is_constructible_v<E, GF> is true.
//
// Effects: Direct-non-list-initializes unex with std::forward<GF>(e.error()).
//
// Postconditions: has_value() is false.
//
// Throws: Any exception thrown by the initialization of unex.

#include <cassert>
#include <expected>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"
#include "test_macros.h"
#include "../../types.h"

// Test Constraints
static_assert(std::is_constructible_v<std::expected<void, int>, std::unexpected<int>>);
static_assert(std::is_constructible_v<std::expected<void, MoveOnly>, std::unexpected<MoveOnly>>);

// !is_constructible_v<E, GF>
struct foo {};
static_assert(!std::is_constructible_v<std::expected<void, int>, std::unexpected<foo>>);

// explicit(!is_convertible_v<G, E>)
struct NotConvertible {
  explicit NotConvertible(int);
};
static_assert(std::is_convertible_v<std::unexpected<int>&&, std::expected<void, int>>);
static_assert(!std::is_convertible_v<std::unexpected<int>&&, std::expected<void, NotConvertible>>);

struct MyInt {
  int i;
  constexpr MyInt(int ii) : i(ii) {}
  friend constexpr bool operator==(const MyInt&, const MyInt&) = default;
};

template <class Err>
constexpr void testInt() {
  std::unexpected<int> u(5);
  std::expected<void, Err> e(std::move(u));
  assert(!e.has_value());
  assert(e.error() == 5);
}

constexpr void testMoveOnly() {
  std::unexpected<MoveOnly> u(MoveOnly(5));
  std::expected<void, MoveOnly> e(std::move(u));
  assert(!e.has_value());
  assert(e.error() == 5);
  assert(u.error() == 0);
}

constexpr bool test() {
  testInt<int>();
  testInt<MyInt>();
  testInt<MoveOnly>();
  testInt<TailClobberer<1>>();
  testMoveOnly();
  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Throwing {
    Throwing(int) { throw Except{}; }
  };

  {
    std::unexpected<int> u(5);
    try {
      [[maybe_unused]] std::expected<void, Throwing> e(std::move(u));
      assert(false);
    } catch (Except) {
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
