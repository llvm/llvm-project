//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class U, class G>
//   constexpr explicit(!is_convertible_v<G, E>) expected(expected<U, G>&& rhs);
//
// Let GF be G
//
// Constraints:
// - is_void_v<U> is true; and
// - is_constructible_v<E, GF> is true; and
// - is_constructible_v<unexpected<E>, expected<U, G>&> is false; and
// - is_constructible_v<unexpected<E>, expected<U, G>> is false; and
// - is_constructible_v<unexpected<E>, const expected<U, G>&> is false; and
// - is_constructible_v<unexpected<E>, const expected<U, G>> is false.
//
// Effects: If rhs.has_value() is false, direct-non-list-initializes unex with std::forward<GF>(rhs.error()).
//
// Postconditions: rhs.has_value() is unchanged; rhs.has_value() == this->has_value() is true.
//
// Throws: Any exception thrown by the initialization of unex.

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"
#include "test_macros.h"
#include "../../types.h"

// Test Constraints:
template <class T1, class Err1, class T2, class Err2>
concept canCstrFromExpected = std::is_constructible_v<std::expected<T1, Err1>, std::expected<T2, Err2>&&>;

struct CtorFromInt {
  CtorFromInt(int);
};

static_assert(canCstrFromExpected<void, CtorFromInt, void, int>);

struct NoCtorFromInt {};

// !is_void_v<U>
static_assert(!canCstrFromExpected<void, int, int, int>);

// !is_constructible_v<E, GF>
static_assert(!canCstrFromExpected<void, NoCtorFromInt, void, int>);

template <class T>
struct CtorFrom {
  explicit CtorFrom(int)
    requires(!std::same_as<T, int>);
  explicit CtorFrom(T);
  explicit CtorFrom(auto&&) = delete;
};

// Note for below 4 tests, because their E is constructible from cvref of std::expected<void, int>,
// unexpected<E> will be constructible from cvref of std::expected<void, int>
// is_constructible_v<unexpected<E>, expected<U, G>&>
static_assert(!canCstrFromExpected<void, CtorFrom<std::expected<void, int>&>, void, int>);

// is_constructible_v<unexpected<E>, expected<U, G>>
static_assert(!canCstrFromExpected<void, CtorFrom<std::expected<void, int>&&>, void, int>);

// is_constructible_v<unexpected<E>, const expected<U, G>&> is false
static_assert(!canCstrFromExpected<void, CtorFrom<std::expected<void, int> const&>, void, int>);

// is_constructible_v<unexpected<E>, const expected<U, G>>
static_assert(!canCstrFromExpected<void, CtorFrom<std::expected<void, int> const&&>, void, int>);

// test explicit
static_assert(std::is_convertible_v<std::expected<void, int>&&, std::expected<void, long>>);

// !is_convertible_v<GF, E>.
static_assert(std::is_constructible_v<std::expected<void, CtorFrom<int>>, std::expected<void, int>&&>);
static_assert(!std::is_convertible_v<std::expected<void, int>&&, std::expected<void, CtorFrom<int>>>);

struct Data {
  MoveOnly data;
  constexpr Data(MoveOnly&& m) : data(std::move(m)) {}
};

constexpr bool test() {
  // convert the error
  {
    std::expected<void, MoveOnly> e1(std::unexpect, 5);
    std::expected<void, Data> e2 = std::move(e1);
    assert(!e2.has_value());
    assert(e2.error().data.get() == 5);
    assert(!e1.has_value());
    assert(e1.error().get() == 0);
  }

  // convert TailClobberer
  {
    std::expected<void, TailClobbererNonTrivialMove<1>> e1(std::unexpect);
    std::expected<void, TailClobberer<1>> e2 = std::move(e1);
    assert(!e2.has_value());
    assert(!e1.has_value());
  }

  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct ThrowingInt {
    ThrowingInt(int) { throw Except{}; }
  };

  // throw on converting error
  {
    const std::expected<void, int> e1(std::unexpect);
    try {
      [[maybe_unused]] std::expected<void, ThrowingInt> e2 = e1;
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
