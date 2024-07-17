//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <variant>

// template <class ...Types> class variant;

// void swap(variant& rhs) noexcept(see below)

#include <cassert>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <variant>

#include "test_convertible.h"
#include "test_macros.h"
#include "variant_test_helpers.h"

struct NotSwappable {};
void swap(NotSwappable&, NotSwappable&) = delete;

struct NotCopyable {
  NotCopyable()                              = default;
  NotCopyable(const NotCopyable&)            = delete;
  NotCopyable& operator=(const NotCopyable&) = delete;
};

struct NotCopyableWithSwap {
  NotCopyableWithSwap()                                      = default;
  NotCopyableWithSwap(const NotCopyableWithSwap&)            = delete;
  NotCopyableWithSwap& operator=(const NotCopyableWithSwap&) = delete;
};
constexpr void swap(NotCopyableWithSwap&, NotCopyableWithSwap) {}

struct NotMoveAssignable {
  NotMoveAssignable()                               = default;
  NotMoveAssignable(NotMoveAssignable&&)            = default;
  NotMoveAssignable& operator=(NotMoveAssignable&&) = delete;
};

struct NotMoveAssignableWithSwap {
  NotMoveAssignableWithSwap()                                       = default;
  NotMoveAssignableWithSwap(NotMoveAssignableWithSwap&&)            = default;
  NotMoveAssignableWithSwap& operator=(NotMoveAssignableWithSwap&&) = delete;
};
constexpr void swap(NotMoveAssignableWithSwap&, NotMoveAssignableWithSwap&) noexcept {}

template <bool Throws>
constexpr void do_throw() {}

template <>
void do_throw<true>() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  throw 42;
#else
  std::abort();
#endif
}

template <bool NT_Copy, bool NT_Move, bool NT_CopyAssign, bool NT_MoveAssign, bool NT_Swap, bool EnableSwap = true>
struct NothrowTypeImp {
  int value;
  int* move_called;
  int* move_assign_called;
  int* swap_called;

  constexpr NothrowTypeImp(int v, int* mv_ctr, int* mv_assign, int* swap)
      : value(v), move_called(mv_ctr), move_assign_called(mv_assign), swap_called(swap) {}

  NothrowTypeImp(const NothrowTypeImp& o) noexcept(NT_Copy) : value(o.value) { assert(false); } // never called by test

  constexpr NothrowTypeImp(NothrowTypeImp&& o) noexcept(NT_Move)
      : value(o.value),
        move_called(o.move_called),
        move_assign_called(o.move_assign_called),
        swap_called(o.swap_called) {
    ++*move_called;
    do_throw<!NT_Move>();
    o.value = -1;
  }

  NothrowTypeImp& operator=(const NothrowTypeImp&) noexcept(NT_CopyAssign) {
    assert(false);
    return *this;
  } // never called by the tests

  constexpr NothrowTypeImp& operator=(NothrowTypeImp&& o) noexcept(NT_MoveAssign) {
    ++*move_assign_called;
    do_throw<!NT_MoveAssign>();
    value   = o.value;
    o.value = -1;
    return *this;
  }
};

template <bool NT_Copy, bool NT_Move, bool NT_CopyAssign, bool NT_MoveAssign, bool NT_Swap>
constexpr void
swap(NothrowTypeImp<NT_Copy, NT_Move, NT_CopyAssign, NT_MoveAssign, NT_Swap, true>& lhs,
     NothrowTypeImp<NT_Copy, NT_Move, NT_CopyAssign, NT_MoveAssign, NT_Swap, true>& rhs) noexcept(NT_Swap) {
  ++*lhs.swap_called;
  do_throw<!NT_Swap>();
  std::swap(lhs.value, rhs.value);
}

// throwing copy, nothrow move ctor/assign, no swap provided
using NothrowMoveable = NothrowTypeImp<false, true, false, true, false, false>;
// throwing copy and move assign, nothrow move ctor, no swap provided
using NothrowMoveCtor = NothrowTypeImp<false, true, false, false, false, false>;
// nothrow move ctor, throwing move assignment, swap provided
using NothrowMoveCtorWithThrowingSwap = NothrowTypeImp<false, true, false, false, false, true>;
// throwing move ctor, nothrow move assignment, no swap provided
using ThrowingMoveCtor = NothrowTypeImp<false, false, false, true, false, false>;
// throwing special members, nothrowing swap
using ThrowingTypeWithNothrowSwap = NothrowTypeImp<false, false, false, false, true, true>;
using NothrowTypeWithThrowingSwap = NothrowTypeImp<true, true, true, true, false, true>;
// throwing move assign with nothrow move and nothrow swap
using ThrowingMoveAssignNothrowMoveCtorWithSwap = NothrowTypeImp<false, true, false, false, true, true>;
// throwing move assign with nothrow move but no swap.
using ThrowingMoveAssignNothrowMoveCtor = NothrowTypeImp<false, true, false, false, false, false>;

struct NonThrowingNonNoexceptType {
  int value;
  int* move_called;
  constexpr NonThrowingNonNoexceptType(int v, int* mv_called) : value(v), move_called(mv_called) {}
  constexpr NonThrowingNonNoexceptType(NonThrowingNonNoexceptType&& o) noexcept(false)
      : value(o.value), move_called(o.move_called) {
    ++*move_called;
    o.value = -1;
  }
  NonThrowingNonNoexceptType& operator=(NonThrowingNonNoexceptType&&) noexcept(false) {
    assert(false); // never called by the tests.
    return *this;
  }
};

struct ThrowsOnSecondMove {
  int value;
  int move_count;
  ThrowsOnSecondMove(int v) : value(v), move_count(0) {}
  ThrowsOnSecondMove(ThrowsOnSecondMove&& o) noexcept(false) : value(o.value), move_count(o.move_count + 1) {
    if (move_count == 2)
      do_throw<true>();
    o.value = -1;
  }
  ThrowsOnSecondMove& operator=(ThrowsOnSecondMove&&) {
    assert(false); // not called by test
    return *this;
  }
};

void test_swap_valueless_by_exception() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using V = std::variant<int, MakeEmptyT>;
  { // both empty
    V v1;
    makeEmpty(v1);
    V v2;
    makeEmpty(v2);
    assert(MakeEmptyT::alive == 0);
    { // member swap
      v1.swap(v2);
      assert(v1.valueless_by_exception());
      assert(v2.valueless_by_exception());
      assert(MakeEmptyT::alive == 0);
    }
    { // non-member swap
      swap(v1, v2);
      assert(v1.valueless_by_exception());
      assert(v2.valueless_by_exception());
      assert(MakeEmptyT::alive == 0);
    }
  }
  { // only one empty
    V v1(42);
    V v2;
    makeEmpty(v2);
    { // member swap
      v1.swap(v2);
      assert(v1.valueless_by_exception());
      assert(std::get<0>(v2) == 42);
      // swap again
      v2.swap(v1);
      assert(v2.valueless_by_exception());
      assert(std::get<0>(v1) == 42);
    }
    { // non-member swap
      swap(v1, v2);
      assert(v1.valueless_by_exception());
      assert(std::get<0>(v2) == 42);
      // swap again
      swap(v1, v2);
      assert(v2.valueless_by_exception());
      assert(std::get<0>(v1) == 42);
    }
  }
#endif
}

TEST_CONSTEXPR_CXX20 void test_swap_same_alternative() {
  {
    using V                = std::variant<ThrowingTypeWithNothrowSwap, int>;
    int move_called        = 0;
    int move_assign_called = 0;
    int swap_called        = 0;
    V v1(std::in_place_index<0>, 42, &move_called, &move_assign_called, &swap_called);
    V v2(std::in_place_index<0>, 100, &move_called, &move_assign_called, &swap_called);
    v1.swap(v2);
    assert(swap_called == 1);
    assert(std::get<0>(v1).value == 100);
    assert(std::get<0>(v2).value == 42);
    swap(v1, v2);
    assert(swap_called == 2);
    assert(std::get<0>(v1).value == 42);
    assert(std::get<0>(v2).value == 100);

    assert(move_called == 0);
    assert(move_assign_called == 0);
  }
  {
    using V                = std::variant<NothrowMoveable, int>;
    int move_called        = 0;
    int move_assign_called = 0;
    int swap_called        = 0;
    V v1(std::in_place_index<0>, 42, &move_called, &move_assign_called, &swap_called);
    V v2(std::in_place_index<0>, 100, &move_called, &move_assign_called, &swap_called);
    v1.swap(v2);
    assert(swap_called == 0);
    assert(move_called == 1);
    assert(move_assign_called == 2);
    assert(std::get<0>(v1).value == 100);
    assert(std::get<0>(v2).value == 42);

    move_called        = 0;
    move_assign_called = 0;
    swap_called        = 0;

    swap(v1, v2);
    assert(swap_called == 0);
    assert(move_called == 1);
    assert(move_assign_called == 2);
    assert(std::get<0>(v1).value == 42);
    assert(std::get<0>(v2).value == 100);
  }
}

void test_swap_same_alternative_throws(){
#ifndef TEST_HAS_NO_EXCEPTIONS
    {using V = std::variant<NothrowTypeWithThrowingSwap, int>;
int move_called        = 0;
int move_assign_called = 0;
int swap_called        = 0;
V v1(std::in_place_index<0>, 42, &move_called, &move_assign_called, &swap_called);
V v2(std::in_place_index<0>, 100, &move_called, &move_assign_called, &swap_called);
try {
  v1.swap(v2);
  assert(false);
} catch (int) {
}
assert(swap_called == 1);
assert(move_called == 0);
assert(move_assign_called == 0);
assert(std::get<0>(v1).value == 42);
assert(std::get<0>(v2).value == 100);
}

{
  using V                = std::variant<ThrowingMoveCtor, int>;
  int move_called        = 0;
  int move_assign_called = 0;
  int swap_called        = 0;
  V v1(std::in_place_index<0>, 42, &move_called, &move_assign_called, &swap_called);
  V v2(std::in_place_index<0>, 100, &move_called, &move_assign_called, &swap_called);
  try {
    v1.swap(v2);
    assert(false);
  } catch (int) {
  }
  assert(move_called == 1); // call threw
  assert(move_assign_called == 0);
  assert(swap_called == 0);
  assert(std::get<0>(v1).value == 42); // throw happened before v1 was moved from
  assert(std::get<0>(v2).value == 100);
}
{
  using V                = std::variant<ThrowingMoveAssignNothrowMoveCtor, int>;
  int move_called        = 0;
  int move_assign_called = 0;
  int swap_called        = 0;
  V v1(std::in_place_index<0>, 42, &move_called, &move_assign_called, &swap_called);
  V v2(std::in_place_index<0>, 100, &move_called, &move_assign_called, &swap_called);
  try {
    v1.swap(v2);
    assert(false);
  } catch (int) {
  }
  assert(move_called == 1);
  assert(move_assign_called == 1); // call threw and didn't complete
  assert(swap_called == 0);
  assert(std::get<0>(v1).value == -1); // v1 was moved from
  assert(std::get<0>(v2).value == 100);
}
#endif
}

TEST_CONSTEXPR_CXX20 void test_swap_different_alternatives() {
  {
    using V                = std::variant<NothrowMoveCtorWithThrowingSwap, int>;
    int move_called        = 0;
    int move_assign_called = 0;
    int swap_called        = 0;
    V v1(std::in_place_index<0>, 42, &move_called, &move_assign_called, &swap_called);
    V v2(std::in_place_index<1>, 100);
    v1.swap(v2);
    assert(swap_called == 0);
    // The libc++ implementation double copies the argument, and not
    // the variant swap is called on.
    LIBCPP_ASSERT(move_called == 1);
    assert(move_called <= 2);
    assert(move_assign_called == 0);
    assert(std::get<1>(v1) == 100);
    assert(std::get<0>(v2).value == 42);

    move_called        = 0;
    move_assign_called = 0;
    swap_called        = 0;

    swap(v1, v2);
    assert(swap_called == 0);
    LIBCPP_ASSERT(move_called == 2);
    assert(move_called <= 2);
    assert(move_assign_called == 0);
    assert(std::get<0>(v1).value == 42);
    assert(std::get<1>(v2) == 100);
  }
}

void test_swap_different_alternatives_throws() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    using V                 = std::variant<ThrowingTypeWithNothrowSwap, NonThrowingNonNoexceptType>;
    int move_called1        = 0;
    int move_assign_called1 = 0;
    int swap_called1        = 0;
    int move_called2        = 0;
    V v1(std::in_place_index<0>, 42, &move_called1, &move_assign_called1, &swap_called1);
    V v2(std::in_place_index<1>, 100, &move_called2);
    try {
      v1.swap(v2);
      assert(false);
    } catch (int) {
    }
    assert(swap_called1 == 0);
    assert(move_called1 == 1); // throws
    assert(move_assign_called1 == 0);
    // FIXME: libc++ shouldn't move from T2 here.
    LIBCPP_ASSERT(move_called2 == 1);
    assert(move_called2 <= 1);
    assert(std::get<0>(v1).value == 42);
    if (move_called2 != 0)
      assert(v2.valueless_by_exception());
    else
      assert(std::get<1>(v2).value == 100);
  }
  {
    using V                 = std::variant<NonThrowingNonNoexceptType, ThrowingTypeWithNothrowSwap>;
    int move_called1        = 0;
    int move_called2        = 0;
    int move_assign_called2 = 0;
    int swap_called2        = 0;
    V v1(std::in_place_index<0>, 42, &move_called1);
    V v2(std::in_place_index<1>, 100, &move_called2, &move_assign_called2, &swap_called2);
    try {
      v1.swap(v2);
      assert(false);
    } catch (int) {
    }
    LIBCPP_ASSERT(move_called1 == 0);
    assert(move_called1 <= 1);
    assert(swap_called2 == 0);
    assert(move_called2 == 1); // throws
    assert(move_assign_called2 == 0);
    if (move_called1 != 0)
      assert(v1.valueless_by_exception());
    else
      assert(std::get<0>(v1).value == 42);
    assert(std::get<1>(v2).value == 100);
  }
// FIXME: The tests below are just very libc++ specific
#  ifdef _LIBCPP_VERSION
  {
    using V         = std::variant<ThrowsOnSecondMove, NonThrowingNonNoexceptType>;
    int move_called = 0;
    V v1(std::in_place_index<0>, 42);
    V v2(std::in_place_index<1>, 100, &move_called);
    v1.swap(v2);
    assert(move_called == 2);
    assert(std::get<1>(v1).value == 100);
    assert(std::get<0>(v2).value == 42);
    assert(std::get<0>(v2).move_count == 1);
  }
  {
    using V         = std::variant<NonThrowingNonNoexceptType, ThrowsOnSecondMove>;
    int move_called = 0;
    V v1(std::in_place_index<0>, 42, &move_called);
    V v2(std::in_place_index<1>, 100);
    try {
      v1.swap(v2);
      assert(false);
    } catch (int) {
    }
    assert(move_called == 1);
    assert(v1.valueless_by_exception());
    assert(std::get<0>(v2).value == 42);
  }
#  endif
  // testing libc++ extension. If either variant stores a nothrow move
  // constructible type v1.swap(v2) provides the strong exception safety
  // guarantee.
#  ifdef _LIBCPP_VERSION
  {
    using V                 = std::variant<ThrowingTypeWithNothrowSwap, NothrowMoveable>;
    int move_called1        = 0;
    int move_assign_called1 = 0;
    int swap_called1        = 0;
    int move_called2        = 0;
    int move_assign_called2 = 0;
    int swap_called2        = 0;
    V v1(std::in_place_index<0>, 42, &move_called1, &move_assign_called1, &swap_called1);
    V v2(std::in_place_index<1>, 100, &move_called2, &move_assign_called2, &swap_called2);
    try {
      v1.swap(v2);
      assert(false);
    } catch (int) {
    }
    assert(swap_called1 == 0);
    assert(move_called1 == 1);
    assert(move_assign_called1 == 0);
    assert(swap_called2 == 0);
    assert(move_called2 == 2);
    assert(move_assign_called2 == 0);
    assert(std::get<0>(v1).value == 42);
    assert(std::get<1>(v2).value == 100);
    // swap again, but call v2's swap.

    move_called1        = 0;
    move_assign_called1 = 0;
    swap_called1        = 0;
    move_called2        = 0;
    move_assign_called2 = 0;
    swap_called2        = 0;

    try {
      v2.swap(v1);
      assert(false);
    } catch (int) {
    }
    assert(swap_called1 == 0);
    assert(move_called1 == 1);
    assert(move_assign_called1 == 0);
    assert(swap_called2 == 0);
    assert(move_called2 == 2);
    assert(move_assign_called2 == 0);
    assert(std::get<0>(v1).value == 42);
    assert(std::get<1>(v2).value == 100);
  }
#  endif // _LIBCPP_VERSION
#endif
}

template <class Var>
constexpr auto has_swap_member_imp(int) -> decltype(std::declval<Var&>().swap(std::declval<Var&>()), true) {
  return true;
}

template <class Var>
constexpr auto has_swap_member_imp(long) -> bool {
  return false;
}

template <class Var>
constexpr bool has_swap_member() {
  return has_swap_member_imp<Var>(0);
}

constexpr void test_swap_sfinae() {
  {
    // This variant type does not provide either a member or non-member swap
    // but is still swappable via the generic swap algorithm, since the
    // variant is move constructible and move assignable.
    using V = std::variant<int, NotSwappable>;
    LIBCPP_STATIC_ASSERT(!has_swap_member<V>(), "");
    static_assert(std::is_swappable_v<V>, "");
  }
  {
    using V = std::variant<int, NotCopyable>;
    LIBCPP_STATIC_ASSERT(!has_swap_member<V>(), "");
    static_assert(!std::is_swappable_v<V>, "");
  }
  {
    using V = std::variant<int, NotCopyableWithSwap>;
    LIBCPP_STATIC_ASSERT(!has_swap_member<V>(), "");
    static_assert(!std::is_swappable_v<V>, "");
  }
  {
    using V = std::variant<int, NotMoveAssignable>;
    LIBCPP_STATIC_ASSERT(!has_swap_member<V>(), "");
    static_assert(!std::is_swappable_v<V>, "");
  }
}

TEST_CONSTEXPR_CXX20 void test_swap_noexcept() {
  {
    using V = std::variant<int, NothrowMoveable>;
    static_assert(std::is_swappable_v<V> && has_swap_member<V>(), "");
    static_assert(std::is_nothrow_swappable_v<V>, "");
    // instantiate swap
    V v1, v2;
    v1.swap(v2);
    swap(v1, v2);
  }
  {
    using V = std::variant<int, NothrowMoveCtor>;
    static_assert(std::is_swappable_v<V> && has_swap_member<V>(), "");
    static_assert(!std::is_nothrow_swappable_v<V>, "");
    // instantiate swap
    V v1, v2;
    v1.swap(v2);
    swap(v1, v2);
  }
  {
    using V = std::variant<int, ThrowingTypeWithNothrowSwap>;
    static_assert(std::is_swappable_v<V> && has_swap_member<V>(), "");
    static_assert(!std::is_nothrow_swappable_v<V>, "");
    // instantiate swap
    V v1, v2;
    v1.swap(v2);
    swap(v1, v2);
  }
  {
    using V = std::variant<int, ThrowingMoveAssignNothrowMoveCtor>;
    static_assert(std::is_swappable_v<V> && has_swap_member<V>(), "");
    static_assert(!std::is_nothrow_swappable_v<V>, "");
    // instantiate swap
    V v1, v2;
    v1.swap(v2);
    swap(v1, v2);
  }
  {
    using V = std::variant<int, ThrowingMoveAssignNothrowMoveCtorWithSwap>;
    static_assert(std::is_swappable_v<V> && has_swap_member<V>(), "");
    static_assert(std::is_nothrow_swappable_v<V>, "");
    // instantiate swap
    V v1, v2;
    v1.swap(v2);
    swap(v1, v2);
  }
  {
    using V = std::variant<int, NotMoveAssignableWithSwap>;
    static_assert(std::is_swappable_v<V> && has_swap_member<V>(), "");
    static_assert(std::is_nothrow_swappable_v<V>, "");
    // instantiate swap
    V v1, v2;
    v1.swap(v2);
    swap(v1, v2);
  }
  {
    // This variant type does not provide either a member or non-member swap
    // but is still swappable via the generic swap algorithm, since the
    // variant is move constructible and move assignable.
    using V = std::variant<int, NotSwappable>;
    LIBCPP_STATIC_ASSERT(!has_swap_member<V>(), "");
    static_assert(std::is_swappable_v<V>, "");
    static_assert(std::is_nothrow_swappable_v<V>, "");
    V v1, v2;
    swap(v1, v2);
  }
}

#ifdef _LIBCPP_VERSION
// This is why variant should SFINAE member swap. :-)
template class std::variant<int, NotSwappable>;
#endif

void non_constexpr_test() {
  test_swap_valueless_by_exception();
  test_swap_same_alternative_throws();
  test_swap_different_alternatives_throws();
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_swap_same_alternative();
  test_swap_different_alternatives();
  test_swap_sfinae();
  test_swap_noexcept();

  return true;
}

int main(int, char**) {
  non_constexpr_test();
  test();

#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
