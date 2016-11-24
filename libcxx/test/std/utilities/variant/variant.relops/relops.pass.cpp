// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <variant>

// template <class ...Types>
// constexpr bool
// operator==(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator!=(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator<(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator>(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator<=(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator>=(variant<Types...> const&, variant<Types...> const&) noexcept;

#include <cassert>
#include <type_traits>
#include <utility>
#include <variant>

#include "test_macros.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
struct MakeEmptyT {
  MakeEmptyT() = default;
  MakeEmptyT(MakeEmptyT &&) { throw 42; }
  MakeEmptyT &operator=(MakeEmptyT &&) { throw 42; }
};
inline bool operator==(MakeEmptyT const &, MakeEmptyT const &) {
  assert(false);
  return false;
}
inline bool operator!=(MakeEmptyT const &, MakeEmptyT const &) {
  assert(false);
  return false;
}
inline bool operator<(MakeEmptyT const &, MakeEmptyT const &) {
  assert(false);
  return false;
}
inline bool operator<=(MakeEmptyT const &, MakeEmptyT const &) {
  assert(false);
  return false;
}
inline bool operator>(MakeEmptyT const &, MakeEmptyT const &) {
  assert(false);
  return false;
}
inline bool operator>=(MakeEmptyT const &, MakeEmptyT const &) {
  assert(false);
  return false;
}

template <class Variant> void makeEmpty(Variant &v) {
  Variant v2(std::in_place_type<MakeEmptyT>);
  try {
    v = std::move(v2);
    assert(false);
  } catch (...) {
    assert(v.valueless_by_exception());
  }
}
#endif // TEST_HAS_NO_EXCEPTIONS

void test_equality() {
  {
    using V = std::variant<int, long>;
    constexpr V v1(42);
    constexpr V v2(42);
    static_assert(v1 == v2, "");
    static_assert(v2 == v1, "");
    static_assert(!(v1 != v2), "");
    static_assert(!(v2 != v1), "");
  }
  {
    using V = std::variant<int, long>;
    constexpr V v1(42);
    constexpr V v2(43);
    static_assert(!(v1 == v2), "");
    static_assert(!(v2 == v1), "");
    static_assert(v1 != v2, "");
    static_assert(v2 != v1, "");
  }
  {
    using V = std::variant<int, long>;
    constexpr V v1(42);
    constexpr V v2(42l);
    static_assert(!(v1 == v2), "");
    static_assert(!(v2 == v1), "");
    static_assert(v1 != v2, "");
    static_assert(v2 != v1, "");
  }
  {
    using V = std::variant<int, long>;
    constexpr V v1(42l);
    constexpr V v2(42l);
    static_assert(v1 == v2, "");
    static_assert(v2 == v1, "");
    static_assert(!(v1 != v2), "");
    static_assert(!(v2 != v1), "");
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    using V = std::variant<int, MakeEmptyT>;
    V v1;
    V v2;
    makeEmpty(v2);
    assert(!(v1 == v2));
    assert(!(v2 == v1));
    assert(v1 != v2);
    assert(v2 != v1);
  }
  {
    using V = std::variant<int, MakeEmptyT>;
    V v1;
    makeEmpty(v1);
    V v2;
    assert(!(v1 == v2));
    assert(!(v2 == v1));
    assert(v1 != v2);
    assert(v2 != v1);
  }
  {
    using V = std::variant<int, MakeEmptyT>;
    V v1;
    makeEmpty(v1);
    V v2;
    makeEmpty(v2);
    assert(v1 == v2);
    assert(v2 == v1);
    assert(!(v1 != v2));
    assert(!(v2 != v1));
  }
#endif
}

template <class Var>
constexpr bool test_less(Var const &l, Var const &r, bool expect_less,
                         bool expect_greater) {
  return ((l < r) == expect_less) && (!(l >= r) == expect_less) &&
         ((l > r) == expect_greater) && (!(l <= r) == expect_greater);
}

void test_relational() {
  { // same index, same value
    using V = std::variant<int, long>;
    constexpr V v1(1);
    constexpr V v2(1);
    static_assert(test_less(v1, v2, false, false), "");
  }
  { // same index, value < other_value
    using V = std::variant<int, long>;
    constexpr V v1(0);
    constexpr V v2(1);
    static_assert(test_less(v1, v2, true, false), "");
  }
  { // same index, value > other_value
    using V = std::variant<int, long>;
    constexpr V v1(1);
    constexpr V v2(0);
    static_assert(test_less(v1, v2, false, true), "");
  }
  { // LHS.index() < RHS.index()
    using V = std::variant<int, long>;
    constexpr V v1(0);
    constexpr V v2(0l);
    static_assert(test_less(v1, v2, true, false), "");
  }
  { // LHS.index() > RHS.index()
    using V = std::variant<int, long>;
    constexpr V v1(0l);
    constexpr V v2(0);
    static_assert(test_less(v1, v2, false, true), "");
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  { // LHS.index() < RHS.index(), RHS is empty
    using V = std::variant<int, MakeEmptyT>;
    V v1;
    V v2;
    makeEmpty(v2);
    assert(test_less(v1, v2, false, true));
  }
  { // LHS.index() > RHS.index(), LHS is empty
    using V = std::variant<int, MakeEmptyT>;
    V v1;
    makeEmpty(v1);
    V v2;
    assert(test_less(v1, v2, true, false));
  }
  { // LHS.index() == RHS.index(), LHS and RHS are empty
    using V = std::variant<int, MakeEmptyT>;
    V v1;
    makeEmpty(v1);
    V v2;
    makeEmpty(v2);
    assert(test_less(v1, v2, false, false));
  }
#endif
}

int main() {
  test_equality();
  test_relational();
}
