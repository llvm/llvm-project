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

// constexpr variant(variant const&);

#include <cassert>
#include <type_traits>
#include <variant>

#include "test_macros.h"
#include "test_workarounds.h"

struct NonT {
  constexpr NonT(int v) : value(v) {}
  constexpr NonT(const NonT& o) : value(o.value) {}
  int value;
};
static_assert(!std::is_trivially_copy_constructible<NonT>::value, "");

struct NoCopy {
  NoCopy(const NoCopy&) = delete;
};

struct MoveOnly {
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&)      = default;
};

struct MoveOnlyNT {
  MoveOnlyNT(const MoveOnlyNT&) = delete;
  MoveOnlyNT(MoveOnlyNT&&) {}
};

struct NTCopy {
  constexpr NTCopy(int v) : value(v) {}
  NTCopy(const NTCopy& that) : value(that.value) {}
  NTCopy(NTCopy&&) = delete;
  int value;
};

static_assert(!std::is_trivially_copy_constructible<NTCopy>::value, "");
static_assert(std::is_copy_constructible<NTCopy>::value, "");

struct TCopy {
  constexpr TCopy(int v) : value(v) {}
  TCopy(TCopy const&) = default;
  TCopy(TCopy&&)      = delete;
  int value;
};

static_assert(std::is_trivially_copy_constructible<TCopy>::value, "");

struct TCopyNTMove {
  constexpr TCopyNTMove(int v) : value(v) {}
  TCopyNTMove(const TCopyNTMove&) = default;
  TCopyNTMove(TCopyNTMove&& that) : value(that.value) { that.value = -1; }
  int value;
};

static_assert(std::is_trivially_copy_constructible<TCopyNTMove>::value, "");

#ifndef TEST_HAS_NO_EXCEPTIONS
struct MakeEmptyT {
  static int alive;
  MakeEmptyT() { ++alive; }
  MakeEmptyT(const MakeEmptyT&) {
    ++alive;
    // Don't throw from the copy constructor since variant's assignment
    // operator performs a copy before committing to the assignment.
  }
  MakeEmptyT(MakeEmptyT&&) { throw 42; }
  MakeEmptyT& operator=(const MakeEmptyT&) { throw 42; }
  MakeEmptyT& operator=(MakeEmptyT&&) { throw 42; }
  ~MakeEmptyT() { --alive; }
};

int MakeEmptyT::alive = 0;

template <class Variant>
void makeEmpty(Variant& v) {
  Variant v2(std::in_place_type<MakeEmptyT>);
  try {
    v = std::move(v2);
    assert(false);
  } catch (...) {
    assert(v.valueless_by_exception());
  }
}
#endif // TEST_HAS_NO_EXCEPTIONS

constexpr void test_copy_ctor_sfinae() {
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

  // Make sure we properly propagate triviality (see P0602R4).
  {
    using V = std::variant<int, long>;
    static_assert(std::is_trivially_copy_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, NTCopy>;
    static_assert(!std::is_trivially_copy_constructible<V>::value, "");
    static_assert(std::is_copy_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, TCopy>;
    static_assert(std::is_trivially_copy_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, TCopyNTMove>;
    static_assert(std::is_trivially_copy_constructible<V>::value, "");
  }
}

TEST_CONSTEXPR_CXX20 void test_copy_ctor_basic() {
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

  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  {
    constexpr std::variant<int> v(std::in_place_index<0>, 42);
    static_assert(v.index() == 0, "");
    constexpr std::variant<int> v2 = v;
    static_assert(v2.index() == 0, "");
    static_assert(std::get<0>(v2) == 42, "");
  }
  {
    constexpr std::variant<int, long> v(std::in_place_index<1>, 42);
    static_assert(v.index() == 1, "");
    constexpr std::variant<int, long> v2 = v;
    static_assert(v2.index() == 1, "");
    static_assert(std::get<1>(v2) == 42, "");
  }
  {
    constexpr std::variant<TCopy> v(std::in_place_index<0>, 42);
    static_assert(v.index() == 0, "");
    constexpr std::variant<TCopy> v2(v);
    static_assert(v2.index() == 0, "");
    static_assert(std::get<0>(v2).value == 42, "");
  }
  {
    constexpr std::variant<int, TCopy> v(std::in_place_index<1>, 42);
    static_assert(v.index() == 1, "");
    constexpr std::variant<int, TCopy> v2(v);
    static_assert(v2.index() == 1, "");
    static_assert(std::get<1>(v2).value == 42, "");
  }
  {
    constexpr std::variant<TCopyNTMove> v(std::in_place_index<0>, 42);
    static_assert(v.index() == 0, "");
    constexpr std::variant<TCopyNTMove> v2(v);
    static_assert(v2.index() == 0, "");
    static_assert(std::get<0>(v2).value == 42, "");
  }
  {
    constexpr std::variant<int, TCopyNTMove> v(std::in_place_index<1>, 42);
    static_assert(v.index() == 1, "");
    constexpr std::variant<int, TCopyNTMove> v2(v);
    static_assert(v2.index() == 1, "");
    static_assert(std::get<1>(v2).value == 42, "");
  }
}

void test_copy_ctor_valueless_by_exception() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using V = std::variant<int, MakeEmptyT>;
  V v1;
  makeEmpty(v1);
  const V& cv1 = v1;
  V v(cv1);
  assert(v.valueless_by_exception());
#endif // TEST_HAS_NO_EXCEPTIONS
}

template <std::size_t Idx, class T>
constexpr void test_constexpr_copy_ctor_imp(const T& v) {
  auto v2 = v;
  assert(v2.index() == v.index());
  assert(v2.index() == Idx);
  assert(std::get<Idx>(v2) == std::get<Idx>(v));
}

constexpr void test_constexpr_copy_ctor_trivial() {
  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  using V = std::variant<long, void*, const int>;
#ifdef TEST_WORKAROUND_MSVC_BROKEN_IS_TRIVIALLY_COPYABLE
  static_assert(std::is_trivially_destructible<V>::value, "");
  static_assert(std::is_trivially_copy_constructible<V>::value, "");
  static_assert(std::is_trivially_move_constructible<V>::value, "");
  static_assert(!std::is_copy_assignable<V>::value, "");
  static_assert(!std::is_move_assignable<V>::value, "");
#else  // TEST_WORKAROUND_MSVC_BROKEN_IS_TRIVIALLY_COPYABLE
  static_assert(std::is_trivially_copyable<V>::value, "");
#endif // TEST_WORKAROUND_MSVC_BROKEN_IS_TRIVIALLY_COPYABLE
  static_assert(std::is_trivially_copy_constructible<V>::value, "");
  test_constexpr_copy_ctor_imp<0>(V(42l));
  test_constexpr_copy_ctor_imp<1>(V(nullptr));
  test_constexpr_copy_ctor_imp<2>(V(101));
}

struct NonTrivialCopyCtor {
  int i = 0;
  constexpr NonTrivialCopyCtor(int ii) : i(ii) {}
  constexpr NonTrivialCopyCtor(const NonTrivialCopyCtor& other) : i(other.i) {}
  constexpr NonTrivialCopyCtor(NonTrivialCopyCtor&& other) = default;
  TEST_CONSTEXPR_CXX20 ~NonTrivialCopyCtor()               = default;
  friend constexpr bool operator==(const NonTrivialCopyCtor& x, const NonTrivialCopyCtor& y) { return x.i == y.i; }
};

TEST_CONSTEXPR_CXX20 void test_constexpr_copy_ctor_non_trivial() {
  // Test !is_trivially_move_constructible
  using V = std::variant<long, NonTrivialCopyCtor, void*>;
  static_assert(!std::is_trivially_copy_constructible<V>::value, "");
  test_constexpr_copy_ctor_imp<0>(V(42l));
  test_constexpr_copy_ctor_imp<1>(V(NonTrivialCopyCtor(5)));
  test_constexpr_copy_ctor_imp<2>(V(nullptr));
}

void non_constexpr_test() { test_copy_ctor_valueless_by_exception(); }

constexpr bool cxx17_constexpr_test() {
  test_copy_ctor_sfinae();
  test_constexpr_copy_ctor_trivial();

  return true;
}

TEST_CONSTEXPR_CXX20 bool cxx20_constexpr_test() {
  test_copy_ctor_basic();
  test_constexpr_copy_ctor_non_trivial();

  return true;
}

int main(int, char**) {
  non_constexpr_test();
  cxx17_constexpr_test();
  cxx20_constexpr_test();

  static_assert(cxx17_constexpr_test());
#if TEST_STD_VER >= 20
  static_assert(cxx20_constexpr_test());
#endif
  return 0;
}
