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

// template <class ...Types> class variant;

// variant(variant const&);

#include <cassert>
#include <type_traits>
#include <variant>

#include "test_macros.h"

struct NonT {
  NonT(int v) : value(v) {}
  NonT(NonT const &o) : value(o.value) {}
  int value;
};
static_assert(!std::is_trivially_copy_constructible<NonT>::value, "");

struct NoCopy {
  NoCopy(NoCopy const &) = delete;
};

struct MoveOnly {
  MoveOnly(MoveOnly const &) = delete;
  MoveOnly(MoveOnly &&) = default;
};

struct MoveOnlyNT {
  MoveOnlyNT(MoveOnlyNT const &) = delete;
  MoveOnlyNT(MoveOnlyNT &&) {}
};

#ifndef TEST_HAS_NO_EXCEPTIONS
struct MakeEmptyT {
  static int alive;
  MakeEmptyT() { ++alive; }
  MakeEmptyT(MakeEmptyT const &) {
    ++alive;
    // Don't throw from the copy constructor since variant's assignment
    // operator performs a copy before committing to the assignment.
  }
  MakeEmptyT(MakeEmptyT &&) { throw 42; }
  MakeEmptyT &operator=(MakeEmptyT const &) { throw 42; }
  MakeEmptyT &operator=(MakeEmptyT &&) { throw 42; }
  ~MakeEmptyT() { --alive; }
};

int MakeEmptyT::alive = 0;

template <class Variant> void makeEmpty(Variant &v) {
  Variant v2(std::in_place_type<MakeEmptyT>);
  try {
    v = v2;
    assert(false);
  } catch (...) {
    assert(v.valueless_by_exception());
  }
}
#endif // TEST_HAS_NO_EXCEPTIONS

void test_copy_ctor_sfinae() {
  {
    using V = std::variant<int, long>;
    static_assert(std::is_copy_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, NoCopy>;
    static_assert(!std::is_copy_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, MoveOnly>;
    static_assert(!std::is_copy_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, MoveOnlyNT>;
    static_assert(!std::is_copy_constructible<V>::value, "");
  }
}

void test_copy_ctor_basic() {
  {
    std::variant<int> v(std::in_place_index<0>, 42);
    std::variant<int> v2 = v;
    assert(v2.index() == 0);
    assert(std::get<0>(v2) == 42);
  }
  {
    std::variant<int, long> v(std::in_place_index<1>, 42);
    std::variant<int, long> v2 = v;
    assert(v2.index() == 1);
    assert(std::get<1>(v2) == 42);
  }
  {
    std::variant<NonT> v(std::in_place_index<0>, 42);
    assert(v.index() == 0);
    std::variant<NonT> v2(v);
    assert(v2.index() == 0);
    assert(std::get<0>(v2).value == 42);
  }
  {
    std::variant<int, NonT> v(std::in_place_index<1>, 42);
    assert(v.index() == 1);
    std::variant<int, NonT> v2(v);
    assert(v2.index() == 1);
    assert(std::get<1>(v2).value == 42);
  }
}

void test_copy_ctor_valueless_by_exception() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using V = std::variant<int, MakeEmptyT>;
  V v1;
  makeEmpty(v1);
  V const &cv1 = v1;
  V v(cv1);
  assert(v.valueless_by_exception());
#endif
}

template <size_t Idx>
constexpr bool test_constexpr_copy_ctor_extension_imp(
    std::variant<long, void*, const int> const& v)
{
  auto v2 = v;
  return v2.index() == v.index() &&
         v2.index() == Idx &&
        std::get<Idx>(v2) == std::get<Idx>(v);
}

void test_constexpr_copy_ctor_extension() {
#ifdef _LIBCPP_VERSION
  using V = std::variant<long, void*, const int>;
  static_assert(std::is_trivially_copyable<V>::value, "");
  static_assert(std::is_trivially_copy_constructible<V>::value, "");
  static_assert(test_constexpr_copy_ctor_extension_imp<0>(V(42l)), "");
  static_assert(test_constexpr_copy_ctor_extension_imp<1>(V(nullptr)), "");
  static_assert(test_constexpr_copy_ctor_extension_imp<2>(V(101)), "");
#endif
}

int main() {
  test_copy_ctor_basic();
  test_copy_ctor_valueless_by_exception();
  test_copy_ctor_sfinae();
  test_constexpr_copy_ctor_extension();
}
