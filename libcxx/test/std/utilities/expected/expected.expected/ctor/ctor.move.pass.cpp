//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr expected(expected&& rhs) noexcept(see below);
//
// Constraints:
// - is_move_constructible_v<T> is true and
// - is_move_constructible_v<E> is true.
//
// Effects: If rhs.has_value() is true, direct-non-list-initializes val with std::move(*rhs).
// Otherwise, direct-non-list-initializes unex with std::move(rhs.error()).
//
// Postconditions: rhs.has_value() is unchanged; rhs.has_value() == this->has_value() is true.
//
// Throws: Any exception thrown by the initialization of val or unex.
//
// Remarks: The exception specification is equivalent to is_nothrow_move_constructible_v<T> && is_nothrow_move_constructible_v<E>.
//
// This constructor is trivial if
// - is_trivially_move_constructible_v<T> is true and
// - is_trivially_move_constructible_v<E> is true.

#include <cassert>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "../../types.h"

struct NonMovable {
  NonMovable(NonMovable&&) = delete;
};

struct MovableNonTrivial {
  int i;
  constexpr MovableNonTrivial(int ii) : i(ii) {}
  constexpr MovableNonTrivial(MovableNonTrivial&& o) : i(o.i) { o.i = 0; }
  friend constexpr bool operator==(const MovableNonTrivial&, const MovableNonTrivial&) = default;
};

struct MoveMayThrow {
  MoveMayThrow(MoveMayThrow&&) {}
};

// Test Constraints:
// - is_move_constructible_v<T> is true and
// - is_move_constructible_v<E> is true.
static_assert(std::is_move_constructible_v<std::expected<int, int>>);
static_assert(std::is_move_constructible_v<std::expected<MovableNonTrivial, int>>);
static_assert(std::is_move_constructible_v<std::expected<int, MovableNonTrivial>>);
static_assert(std::is_move_constructible_v<std::expected<MovableNonTrivial, MovableNonTrivial>>);
static_assert(!std::is_move_constructible_v<std::expected<NonMovable, int>>);
static_assert(!std::is_move_constructible_v<std::expected<int, NonMovable>>);
static_assert(!std::is_move_constructible_v<std::expected<NonMovable, NonMovable>>);

// Test: This constructor is trivial if
// - is_trivially_move_constructible_v<T> is true and
// - is_trivially_move_constructible_v<E> is true.
static_assert(std::is_trivially_move_constructible_v<std::expected<int, int>>);
static_assert(!std::is_trivially_move_constructible_v<std::expected<MovableNonTrivial, int>>);
static_assert(!std::is_trivially_move_constructible_v<std::expected<int, MovableNonTrivial>>);
static_assert(!std::is_trivially_move_constructible_v<std::expected<MovableNonTrivial, MovableNonTrivial>>);

// Test: The exception specification is equivalent to
// is_nothrow_move_constructible_v<T> && is_nothrow_move_constructible_v<E>.
static_assert(std::is_nothrow_move_constructible_v<std::expected<int, int>>);
static_assert(!std::is_nothrow_move_constructible_v<std::expected<MoveMayThrow, int>>);
static_assert(!std::is_nothrow_move_constructible_v<std::expected<int, MoveMayThrow>>);
static_assert(!std::is_nothrow_move_constructible_v<std::expected<MoveMayThrow, MoveMayThrow>>);

constexpr bool test() {
  // move the value non-trivial
  {
    std::expected<MovableNonTrivial, int> e1(5);
    auto e2 = std::move(e1);
    assert(e2.has_value());
    assert(e2.value().i == 5);
    assert(e1.has_value());
    assert(e1.value().i == 0);
  }

  // move the error non-trivial
  {
    std::expected<int, MovableNonTrivial> e1(std::unexpect, 5);
    auto e2 = std::move(e1);
    assert(!e2.has_value());
    assert(e2.error().i == 5);
    assert(!e1.has_value());
    assert(e1.error().i == 0);
  }

  // move the value trivial
  {
    std::expected<int, int> e1(5);
    auto e2 = std::move(e1);
    assert(e2.has_value());
    assert(e2.value() == 5);
    assert(e1.has_value());
  }

  // move the error trivial
  {
    std::expected<int, int> e1(std::unexpect, 5);
    auto e2 = std::move(e1);
    assert(!e2.has_value());
    assert(e2.error() == 5);
    assert(!e1.has_value());
  }

  // move TailClobbererNonTrivialMove as value
  {
    std::expected<TailClobbererNonTrivialMove<0>, bool> e1;
    auto e2 = std::move(e1);
    assert(e2.has_value());
    assert(e1.has_value());
  }

  // move TailClobbererNonTrivialMove as error
  {
    std::expected<bool, TailClobbererNonTrivialMove<1>> e1(std::unexpect);
    auto e2 = std::move(e1);
    assert(!e2.has_value());
    assert(!e1.has_value());
  }

  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Throwing {
    Throwing() = default;
    Throwing(Throwing&&) { throw Except{}; }
  };

  // throw on moving value
  {
    std::expected<Throwing, int> e1;
    try {
      [[maybe_unused]] auto e2 = std::move(e1);
      assert(false);
    } catch (Except) {
    }
  }

  // throw on moving error
  {
    std::expected<int, Throwing> e1(std::unexpect);
    try {
      [[maybe_unused]] auto e2 = std::move(e1);
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
