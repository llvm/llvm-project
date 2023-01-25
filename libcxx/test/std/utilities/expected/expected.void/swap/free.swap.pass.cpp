//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// Older Clangs do not support the C++20 feature to constrain destructors
// XFAIL: clang-14, clang-15, apple-clang-14

// friend constexpr void swap(expected& x, expected& y) noexcept(noexcept(swap(x,y)));

#include <cassert>
#include <expected>
#include <type_traits>
#include <utility>

#include "../../types.h"
#include "test_macros.h"

// Test constraint
static_assert(std::is_swappable_v<std::expected<void, int>>);

struct NotSwappable {
  NotSwappable& operator=(const NotSwappable&) = delete;
};
void swap(NotSwappable&, NotSwappable&) = delete;

// !is_swappable_v<E>
static_assert(!std::is_swappable_v<std::expected<void, NotSwappable>>);

struct NotMoveContructible {
  NotMoveContructible(NotMoveContructible&&) = delete;
  friend void swap(NotMoveContructible&, NotMoveContructible&) {}
};

// !is_move_constructible_v<E>
static_assert(!std::is_swappable_v<std::expected<void, NotMoveContructible>>);

// Test noexcept
struct MoveMayThrow {
  MoveMayThrow(MoveMayThrow&&) noexcept(false);
  friend void swap(MoveMayThrow&, MoveMayThrow&) noexcept {}
};

template <class E>
concept FreeSwapNoexcept =
    requires(std::expected<void, E> x, std::expected<void, E> y) {
      { swap(x, y) } noexcept;
    };

static_assert(FreeSwapNoexcept<int>);

// !is_nothrow_move_constructible_v<E>
static_assert(!FreeSwapNoexcept<MoveMayThrow>);

struct SwapMayThrow {
  friend void swap(SwapMayThrow&, SwapMayThrow&) noexcept(false) {}
};

// !is_nothrow_swappable_v<E>
static_assert(!FreeSwapNoexcept<SwapMayThrow>);

constexpr bool test() {
  // this->has_value() && rhs.has_value()
  {
    std::expected<void, int> x;
    std::expected<void, int> y;
    swap(x, y);

    assert(x.has_value());
    assert(y.has_value());
  }

  // !this->has_value() && !rhs.has_value()
  {
    std::expected<void, ADLSwap> x(std::unexpect, 5);
    std::expected<void, ADLSwap> y(std::unexpect, 10);
    swap(x, y);

    assert(!x.has_value());
    assert(x.error().i == 10);
    assert(x.error().adlSwapCalled);
    assert(!y.has_value());
    assert(y.error().i == 5);
    assert(y.error().adlSwapCalled);
  }

  // this->has_value() && !rhs.has_value()
  {
    Traced::state s{};
    std::expected<void, Traced> e1(std::in_place);
    std::expected<void, Traced> e2(std::unexpect, s, 10);

    e1.swap(e2);

    assert(!e1.has_value());
    assert(e1.error().data_ == 10);
    assert(e2.has_value());

    assert(s.moveCtorCalled);
    assert(s.dtorCalled);
  }

  // !this->has_value() && rhs.has_value()
  {
    Traced::state s{};
    std::expected<void, Traced> e1(std::unexpect, s, 10);
    std::expected<void, Traced> e2(std::in_place);

    e1.swap(e2);

    assert(e1.has_value());
    assert(!e2.has_value());
    assert(e2.error().data_ == 10);

    assert(s.moveCtorCalled);
    assert(s.dtorCalled);
  }

  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  // !e1.has_value() && e2.has_value()
  {
    bool e1Destroyed = false;
    std::expected<void, ThrowOnMove> e1(std::unexpect, e1Destroyed);
    std::expected<void, ThrowOnMove> e2(std::in_place);
    try {
      swap(e1, e2);
      assert(false);
    } catch (Except) {
      assert(!e1.has_value());
      assert(e2.has_value());
      assert(!e1Destroyed);
    }
  }

  // e1.has_value() && !e2.has_value()
  {
    bool e2Destroyed = false;
    std::expected<void, ThrowOnMove> e1(std::in_place);
    std::expected<void, ThrowOnMove> e2(std::unexpect, e2Destroyed);
    try {
      swap(e1, e2);
      assert(false);
    } catch (Except) {
      assert(e1.has_value());
      assert(!e2.has_value());
      assert(!e2Destroyed);
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
