//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Older Clangs do not support the C++20 feature to constrain destructors
// XFAIL: clang-14, apple-clang-14

// template<class G>
//   constexpr expected& operator=(const unexpected<G>& e);
//
// Let GF be const G&
// Constraints:
// - is_constructible_v<E, GF> is true; and
// - is_assignable_v<E&, GF> is true; and
// - is_nothrow_constructible_v<E, GF> || is_nothrow_move_constructible_v<T> ||
//   is_nothrow_move_constructible_v<E> is true.
//
// Effects:
// - If has_value() is true, equivalent to:
//   reinit-expected(unex, val, std::forward<GF>(e.error()));
//   has_val = false;
// - Otherwise, equivalent to: unex = std::forward<GF>(e.error());
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
static_assert(std::is_assignable_v<std::expected<int, int>&, const std::unexpected<int>&>);

// !is_constructible_v<E, GF>
static_assert(
    !std::is_assignable_v<std::expected<int, NotCopyConstructible>&, const std::unexpected<NotCopyConstructible>&>);

// !is_assignable_v<E&, GF>
static_assert(!std::is_assignable_v<std::expected<int, NotCopyAssignable>&, const std::unexpected<NotCopyAssignable>&>);

template <bool moveNoexcept, bool convertNoexcept>
struct MaybeNoexcept {
  explicit MaybeNoexcept(int) noexcept(convertNoexcept);
  MaybeNoexcept(MaybeNoexcept&&) noexcept(moveNoexcept);
  MaybeNoexcept& operator=(MaybeNoexcept&&) = default;
  MaybeNoexcept& operator=(int);
};

// !is_nothrow_constructible_v<E, GF> && !is_nothrow_move_constructible_v<T> &&
// is_nothrow_move_constructible_v<E>
static_assert(std::is_assignable_v<std::expected<MaybeNoexcept<false, false>, MaybeNoexcept<true, false>>&,
                                   const std::unexpected<int>&>);

// is_nothrow_constructible_v<E, GF> && !is_nothrow_move_constructible_v<T> &&
// !is_nothrow_move_constructible_v<E>
static_assert(std::is_assignable_v<std::expected<MaybeNoexcept<false, false>, MaybeNoexcept<false, true>>&,
                                   const std::unexpected<int>&>);

// !is_nothrow_constructible_v<E, GF> && is_nothrow_move_constructible_v<T> &&
// !is_nothrow_move_constructible_v<E>
static_assert(std::is_assignable_v<std::expected<MaybeNoexcept<true, true>, MaybeNoexcept<false, false>>&,
                                   const std::unexpected<int>&>);

// !is_nothrow_constructible_v<E, GF> && !is_nothrow_move_constructible_v<T> &&
// !is_nothrow_move_constructible_v<E>
static_assert(!std::is_assignable_v<std::expected<MaybeNoexcept<false, false>, MaybeNoexcept<false, false>>&,
                                    const std::unexpected<int>&>);

constexpr bool test() {
  // - If has_value() is true, equivalent to:
  //   reinit-expected(unex, val, std::forward<GF>(e.error()));
  // is_nothrow_constructible_v<E, GF>
  //
  //  In this case, it should call the branch
  //    destroy_at(addressof(oldval));
  //    construct_at(addressof(newval), std::forward<Args>(args)...);
  {
    BothNoexcept::state oldState{};
    std::expected<BothNoexcept, BothNoexcept> e(std::in_place, oldState, 5);
    const std::unexpected<int> un(10);
    decltype(auto) x = (e = un);
    static_assert(std::same_as<decltype(x), std::expected<BothNoexcept, BothNoexcept>&>);
    assert(&x == &e);

    assert(!oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(e.error().copiedFromInt);
  }

  // - If has_value() is true, equivalent to:
  //   reinit-expected(unex, val, std::forward<GF>(e.error()));
  // !is_nothrow_constructible_v<E, GF> && is_nothrow_move_constructible_v<E>
  //
  //  In this case, it should call the branch
  //  T tmp(std::forward<Args>(args)...);
  //  destroy_at(addressof(oldval));
  //  construct_at(addressof(newval), std::move(tmp));
  {
    BothNoexcept::state oldState{};
    std::expected<BothNoexcept, MoveNoexceptConvThrow> e(std::in_place, oldState, 5);
    const std::unexpected<int> un(10);
    decltype(auto) x = (e = un);
    static_assert(std::same_as<decltype(x), std::expected<BothNoexcept, MoveNoexceptConvThrow>&>);
    assert(&x == &e);

    assert(!oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(!e.error().copiedFromInt);
    assert(e.error().movedFromTmp);
  }

  // - If has_value() is true, equivalent to:
  //   reinit-expected(unex, val, std::forward<GF>(e.error()));
  // !is_nothrow_constructible_v<E, GF> && !is_nothrow_move_constructible_v<E>
  // is_nothrow_move_constructible_v<T>
  //
  //  In this case, it should call the branch
  //  U tmp(std::move(oldval));
  //  destroy_at(addressof(oldval));
  //  try {
  //    construct_at(addressof(newval), std::forward<Args>(args)...);
  //  } catch (...) {
  //    construct_at(addressof(oldval), std::move(tmp));
  //    throw;
  //  }
  {
    BothNoexcept::state oldState{};
    std::expected<BothNoexcept, BothMayThrow> e(std::in_place, oldState, 5);
    const std::unexpected<int> un(10);
    decltype(auto) x = (e = un);
    static_assert(std::same_as<decltype(x), std::expected<BothNoexcept, BothMayThrow>&>);
    assert(&x == &e);

    assert(oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(e.error().copiedFromInt);
  }

  // Otherwise, equivalent to: unex = std::forward<GF>(e.error());
  {
    Traced::state oldState{};
    Traced::state newState{};
    std::expected<int, Traced> e1(std::unexpect, oldState, 5);
    const std::unexpected<Traced> e(std::in_place, newState, 10);
    decltype(auto) x = (e1 = e);
    static_assert(std::same_as<decltype(x), std::expected<int, Traced>&>);
    assert(&x == &e1);

    assert(!e1.has_value());
    assert(e1.error().data_ == 10);
    assert(oldState.copyAssignCalled);
  }
  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  std::expected<int, ThrowOnConvert> e1(std::in_place, 5);
  const std::unexpected<int> un(10);
  try {
    e1 = un;
    assert(false);
  } catch (Except) {
    assert(e1.has_value());
    assert(*e1 == 5);
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test();
  static_assert(test());
  testException();
  return 0;
}
