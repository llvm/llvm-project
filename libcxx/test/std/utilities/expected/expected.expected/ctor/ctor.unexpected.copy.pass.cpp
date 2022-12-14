//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class G>
//   constexpr explicit(!is_convertible_v<const G&, E>) expected(const unexpected<G>& e);
//
// Let GF be const G&
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

// Test Constraints
static_assert(std::is_constructible_v<std::expected<int, int>, const std::unexpected<int>&>);

// !is_constructible_v<E, GF>
struct foo {};
static_assert(!std::is_constructible_v<std::expected<int, int>, const std::unexpected<foo>&>);
static_assert(!std::is_constructible_v<std::expected<int, MoveOnly>, const std::unexpected<MoveOnly>&>);

// explicit(!is_convertible_v<const G&, E>)
struct NotConvertible {
  explicit NotConvertible(int);
};
static_assert(std::is_convertible_v<const std::unexpected<int>&, std::expected<int, int>>);
static_assert(!std::is_convertible_v<const std::unexpected<int>&, std::expected<int, NotConvertible>>);

struct MyInt {
  int i;
  constexpr MyInt(int ii) : i(ii) {}
  friend constexpr bool operator==(const MyInt&, const MyInt&) = default;
};

template <class T>
constexpr void testUnexpected() {
  const std::unexpected<int> u(5);
  std::expected<int, T> e(u);
  assert(!e.has_value());
  assert(e.error() == 5);
}

constexpr bool test() {
  testUnexpected<int>();
  testUnexpected<MyInt>();
  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Except {};

  struct Throwing {
    Throwing(int) { throw Except{}; }
  };

  {
    const std::unexpected<int> u(5);
    try {
      [[maybe_unused]] std::expected<int, Throwing> e(u);
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
