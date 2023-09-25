//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// Older Clangs do not support the C++20 feature to constrain destructors
// XFAIL: clang-15

// friend constexpr void swap(expected& x, expected& y) noexcept(noexcept(x.swap(y)));

#include <cassert>
#include <expected>
#include <type_traits>
#include <utility>

#include "../../types.h"
#include "test_macros.h"

// Test Constraints:
struct NotSwappable {
  NotSwappable operator=(const NotSwappable&) = delete;
};
void swap(NotSwappable&, NotSwappable&) = delete;

static_assert(std::is_swappable_v<std::expected<int, int>>);

// !is_swappable_v<T>
static_assert(!std::is_swappable_v<std::expected<NotSwappable, int>>);

// !is_swappable_v<E>
static_assert(!std::is_swappable_v<std::expected<int, NotSwappable>>);

struct NotMoveContructible {
  NotMoveContructible(NotMoveContructible&&) = delete;
  friend void swap(NotMoveContructible&, NotMoveContructible&) {}
};

// !is_move_constructible_v<T>
static_assert(!std::is_swappable_v<std::expected<NotMoveContructible, int>>);

// !is_move_constructible_v<E>
static_assert(!std::is_swappable_v<std::expected<int, NotMoveContructible>>);

struct MoveMayThrow {
  MoveMayThrow(MoveMayThrow&&) noexcept(false);
  friend void swap(MoveMayThrow&, MoveMayThrow&) noexcept {}
};

// !is_nothrow_move_constructible_v<T> && is_nothrow_move_constructible_v<E>
static_assert(std::is_swappable_v<std::expected<MoveMayThrow, int>>);

// is_nothrow_move_constructible_v<T> && !is_nothrow_move_constructible_v<E>
static_assert(std::is_swappable_v<std::expected<int, MoveMayThrow>>);

// !is_nothrow_move_constructible_v<T> && !is_nothrow_move_constructible_v<E>
static_assert(!std::is_swappable_v<std::expected<MoveMayThrow, MoveMayThrow>>);

// Test noexcept
static_assert(std::is_nothrow_swappable_v<std::expected<int, int>>);

// !is_nothrow_move_constructible_v<T>
static_assert(!std::is_nothrow_swappable_v<std::expected<MoveMayThrow, int>>);

// !is_nothrow_move_constructible_v<E>
static_assert(!std::is_nothrow_swappable_v<std::expected<int, MoveMayThrow>>);

struct SwapMayThrow {
  friend void swap(SwapMayThrow&, SwapMayThrow&) noexcept(false) {}
};

// !is_nothrow_swappable_v<T>
static_assert(!std::is_nothrow_swappable_v<std::expected<SwapMayThrow, int>>);

// !is_nothrow_swappable_v<E>
static_assert(!std::is_nothrow_swappable_v<std::expected<int, SwapMayThrow>>);

constexpr bool test() {
  // this->has_value() && rhs.has_value()
  {
    std::expected<ADLSwap, int> x(std::in_place, 5);
    std::expected<ADLSwap, int> y(std::in_place, 10);
    swap(x, y);

    assert(x.has_value());
    assert(x->i == 10);
    assert(x->adlSwapCalled);
    assert(y.has_value());
    assert(y->i == 5);
    assert(y->adlSwapCalled);
  }

  // !this->has_value() && !rhs.has_value()
  {
    std::expected<int, ADLSwap> x(std::unexpect, 5);
    std::expected<int, ADLSwap> y(std::unexpect, 10);
    swap(x, y);

    assert(!x.has_value());
    assert(x.error().i == 10);
    assert(x.error().adlSwapCalled);
    assert(!y.has_value());
    assert(y.error().i == 5);
    assert(y.error().adlSwapCalled);
  }

  // this->has_value() && !rhs.has_value()
  // && is_nothrow_move_constructible_v<E>
  {
    std::expected<TrackedMove<true>, TrackedMove<true>> e1(std::in_place, 5);
    std::expected<TrackedMove<true>, TrackedMove<true>> e2(std::unexpect, 10);

    swap(e1, e2);

    assert(!e1.has_value());
    assert(e1.error().i == 10);
    assert(e2.has_value());
    assert(e2->i == 5);

    assert(e1.error().numberOfMoves == 2);
    assert(!e1.error().swapCalled);
    assert(e2->numberOfMoves == 1);
    assert(!e2->swapCalled);
  }

  // this->has_value() && !rhs.has_value()
  // && !is_nothrow_move_constructible_v<E>
  {
    std::expected<TrackedMove<true>, TrackedMove<false>> e1(std::in_place, 5);
    std::expected<TrackedMove<true>, TrackedMove<false>> e2(std::unexpect, 10);

    e1.swap(e2);

    assert(!e1.has_value());
    assert(e1.error().i == 10);
    assert(e2.has_value());
    assert(e2->i == 5);

    assert(e1.error().numberOfMoves == 1);
    assert(!e1.error().swapCalled);
    assert(e2->numberOfMoves == 2);
    assert(!e2->swapCalled);
  }

  // !this->has_value() && rhs.has_value()
  // && is_nothrow_move_constructible_v<E>
  {
    std::expected<TrackedMove<true>, TrackedMove<true>> e1(std::unexpect, 10);
    std::expected<TrackedMove<true>, TrackedMove<true>> e2(std::in_place, 5);

    swap(e1, e2);

    assert(e1.has_value());
    assert(e1->i == 5);
    assert(!e2.has_value());
    assert(e2.error().i == 10);

    assert(e1->numberOfMoves == 1);
    assert(!e1->swapCalled);
    assert(e2.error().numberOfMoves == 2);
    assert(!e2.error().swapCalled);
  }

  // !this->has_value() && rhs.has_value()
  // && !is_nothrow_move_constructible_v<E>
  {
    std::expected<TrackedMove<true>, TrackedMove<false>> e1(std::unexpect, 10);
    std::expected<TrackedMove<true>, TrackedMove<false>> e2(std::in_place, 5);

    swap(e1, e2);

    assert(e1.has_value());
    assert(e1->i == 5);
    assert(!e2.has_value());
    assert(e2.error().i == 10);

    assert(e1->numberOfMoves == 2);
    assert(!e1->swapCalled);
    assert(e2.error().numberOfMoves == 1);
    assert(!e2.error().swapCalled);
  }

  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  // !e1.has_value() && e2.has_value()
  {
    std::expected<ThrowOnMoveConstruct, int> e1(std::unexpect, 5);
    std::expected<ThrowOnMoveConstruct, int> e2(std::in_place);
    try {
      swap(e1, e2);
      assert(false);
    } catch (Except) {
      assert(!e1.has_value());
      assert(e1.error() == 5);
    }
  }

  // e1.has_value() && !e2.has_value()
  {
    std::expected<int, ThrowOnMoveConstruct> e1(5);
    std::expected<int, ThrowOnMoveConstruct> e2(std::unexpect);
    try {
      swap(e1, e2);
      assert(false);
    } catch (Except) {
      assert(e1.has_value());
      assert(*e1 == 5);
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
