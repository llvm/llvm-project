//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// Older Clangs do not support the C++20 feature to constrain destructors
// XFAIL: apple-clang-14

// template<class G>
//   constexpr expected& operator=(const unexpected<G>& e);
//
// Let GF be const G&
//
// Constraints: is_constructible_v<E, GF> is true and is_assignable_v<E&, GF> is true.
//
// Effects:
// - If has_value() is true, equivalent to:
//   construct_at(addressof(unex), std::forward<GF>(e.error()));
//   has_val = false;
// - Otherwise, equivalent to: unex = std::forward<GF>(e.error());
//
// Returns: *this.

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "../../types.h"
#include "test_macros.h"

struct NotCopyConstructible {
  NotCopyConstructible(const NotCopyConstructible&)            = delete;
  NotCopyConstructible& operator=(const NotCopyConstructible&) = default;
};

struct NotCopyAssignable {
  NotCopyAssignable(const NotCopyAssignable&)            = default;
  NotCopyAssignable& operator=(const NotCopyAssignable&) = delete;
};

struct MoveMayThrow {
  MoveMayThrow(MoveMayThrow const&)            = default;
  MoveMayThrow& operator=(const MoveMayThrow&) = default;
  MoveMayThrow(MoveMayThrow&&) noexcept(false) {}
  MoveMayThrow& operator=(MoveMayThrow&&) noexcept(false) { return *this; }
};

// Test constraints
static_assert(std::is_assignable_v<std::expected<void, int>&, const std::unexpected<int>&>);

// !is_constructible_v<E, GF>
static_assert(
    !std::is_assignable_v<std::expected<void, NotCopyConstructible>&, const std::unexpected<NotCopyConstructible>&>);

// !is_assignable_v<E&, GF>
static_assert(
    !std::is_assignable_v<std::expected<void, NotCopyAssignable>&, const std::unexpected<NotCopyAssignable>&>);

constexpr bool test() {
  // - If has_value() is true, equivalent to:
  //   construct_at(addressof(unex), std::forward<GF>(e.error()));
  //   has_val = false;
  {
    Traced::state state{};
    std::expected<void, Traced> e;
    std::unexpected<Traced> un(std::in_place, state, 5);
    decltype(auto) x = (e = un);
    static_assert(std::same_as<decltype(x), std::expected<void, Traced>&>);
    assert(&x == &e);
    assert(!e.has_value());
    assert(e.error().data_ == 5);

    assert(state.copyCtorCalled);
  }

  // - Otherwise, equivalent to: unex = std::forward<GF>(e.error());
  {
    Traced::state state1{};
    Traced::state state2{};
    std::expected<void, Traced> e(std::unexpect, state1, 5);
    std::unexpected<Traced> un(std::in_place, state2, 10);
    decltype(auto) x = (e = un);
    static_assert(std::same_as<decltype(x), std::expected<void, Traced>&>);
    assert(&x == &e);
    assert(!e.has_value());
    assert(e.error().data_ == 10);

    assert(state1.copyAssignCalled);
  }

  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  std::expected<void, ThrowOnCopyConstruct> e1(std::in_place);
  std::unexpected<ThrowOnCopyConstruct> un(std::in_place);
  try {
    e1 = un;
    assert(false);
  } catch (Except) {
    assert(e1.has_value());
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test();
  static_assert(test());
  testException();
  return 0;
}
