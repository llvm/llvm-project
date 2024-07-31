//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr void swap(expected& rhs) noexcept(see below);
//
// Constraints:
// is_swappable_v<E> is true and is_move_constructible_v<E> is true.
//
// Throws: Any exception thrown by the expressions in the Effects.
//
// Remarks: The exception specification is equivalent to:
// is_nothrow_move_constructible_v<E> && is_nothrow_swappable_v<E>.

#include <cassert>
#include <expected>
#include <type_traits>
#include <utility>

#include "../../types.h"
#include "test_macros.h"

// Test Constraints:
template <class E>
concept HasMemberSwap = requires(std::expected<void, E> x, std::expected<void, E> y) { x.swap(y); };

static_assert(HasMemberSwap<int>);

struct NotSwappable {};
void swap(NotSwappable&, NotSwappable&) = delete;

// !is_swappable_v<E>
static_assert(!HasMemberSwap<NotSwappable>);

struct NotMoveConstructible {
  NotMoveConstructible(NotMoveConstructible&&) = delete;
  friend void swap(NotMoveConstructible&, NotMoveConstructible&) {}
};

// !is_move_constructible_v<E>
static_assert(!HasMemberSwap<NotMoveConstructible>);

// Test noexcept
struct MoveMayThrow {
  MoveMayThrow(MoveMayThrow&&) noexcept(false);
  friend void swap(MoveMayThrow&, MoveMayThrow&) noexcept {}
};

template <class E>
concept MemberSwapNoexcept = //
    requires(std::expected<void, E> x, std::expected<void, E> y) {
      { x.swap(y) } noexcept;
    };

static_assert(MemberSwapNoexcept<int>);

// !is_nothrow_move_constructible_v<E>
static_assert(!MemberSwapNoexcept<MoveMayThrow>);

struct SwapMayThrow {
  friend void swap(SwapMayThrow&, SwapMayThrow&) noexcept(false) {}
};

// !is_nothrow_swappable_v<E>
static_assert(!MemberSwapNoexcept<SwapMayThrow>);

constexpr bool test() {
  // this->has_value() && rhs.has_value()
  {
    std::expected<void, int> x;
    std::expected<void, int> y;
    x.swap(y);

    assert(x.has_value());
    assert(y.has_value());
  }

  // !this->has_value() && !rhs.has_value()
  {
    std::expected<void, ADLSwap> x(std::unexpect, 5);
    std::expected<void, ADLSwap> y(std::unexpect, 10);
    x.swap(y);

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

  // TailClobberer
  {
    std::expected<void, TailClobbererNonTrivialMove<1>> x(std::in_place);
    std::expected<void, TailClobbererNonTrivialMove<1>> y(std::unexpect);

    x.swap(y);

    // The next line would fail if adjusting the "has value" flag happened
    // _before_ constructing the member object inside the `swap`.
    assert(!x.has_value());
    assert(y.has_value());
  }

  // CheckForInvalidWrites
  {
    {
      CheckForInvalidWrites<true, true> x(std::unexpect);
      CheckForInvalidWrites<true, true> y;

      x.swap(y);

      assert(x.check());
      assert(y.check());
    }
    {
      CheckForInvalidWrites<false, true> x(std::unexpect);
      CheckForInvalidWrites<false, true> y;

      x.swap(y);

      assert(x.check());
      assert(y.check());
    }
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
      e1.swap(e2);
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
      e1.swap(e2);
      assert(false);
    } catch (Except) {
      assert(e1.has_value());
      assert(!e2.has_value());
      assert(!e2Destroyed);
    }
  }

  // TailClobberer
  {
    std::expected<void, TailClobbererNonTrivialMove<0, false, true>> x(std::in_place);
    std::expected<void, TailClobbererNonTrivialMove<0, false, true>> y(std::unexpect);
    try {
      x.swap(y);
      assert(false);
    } catch (Except) {
      // This would fail if `TailClobbererNonTrivialMove<0, false, true>`
      // clobbered the flag before throwing the exception.
      assert(x.has_value());
      assert(!y.has_value());
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
