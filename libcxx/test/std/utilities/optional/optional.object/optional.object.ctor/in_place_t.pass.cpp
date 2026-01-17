//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// template <class... Args>
//   constexpr explicit optional(in_place_t, Args&&... args);   // only for the primary template
// template <class Arg>
//   constexpr explicit optional(in_place_t, Arg arg);          // since C++26, only for optional<T&>

#include <cassert>
#include <optional>
#include <type_traits>

#include "test_macros.h"

using std::in_place;
using std::in_place_t;
using std::optional;

class X {
  int i_;
  int j_ = 0;

public:
  X() : i_(0) {}
  X(int i) : i_(i) {}
  X(int i, int j) : i_(i), j_(j) {}

  ~X() {}

  friend bool operator==(const X& x, const X& y) { return x.i_ == y.i_ && x.j_ == y.j_; }
};

class Y {
  int i_;
  int j_ = 0;

public:
  constexpr Y() : i_(0) {}
  constexpr Y(int i) : i_(i) {}
  constexpr Y(int i, int j) : i_(i), j_(j) {}

  friend constexpr bool operator==(const Y& x, const Y& y) { return x.i_ == y.i_ && x.j_ == y.j_; }
};

class Z {
public:
  Z(int) { TEST_THROW(6); }
};

#if TEST_STD_VER >= 26
static_assert(!std::is_constructible_v<optional<const int&>, in_place_t>);
static_assert(std::is_constructible_v<optional<const int&>, in_place_t, int&>);
static_assert(std::is_constructible_v<optional<const int&>, in_place_t, const int&>);
static_assert(!std::is_constructible_v<optional<const int&>, in_place_t, int>);
static_assert(!std::is_constructible_v<optional<const int&>, in_place_t, const int>);
static_assert(!std::is_constructible_v<optional<const int&>, in_place_t, long&>);
static_assert(!std::is_constructible_v<optional<const int&>, in_place_t, const long&>);

// Test that initilization in std::optional<std::initializer_list<T>&>{in_place, il} selects the (in_place_t, Arg&&)
// constructor.
// Otherwise, the created optional would store a dangling reference.
constexpr bool test_ref_initializer_list() {
  std::initializer_list<int> il{4, 2};
  optional<std::initializer_list<int>&> opt{in_place, il};

  auto il2 = opt.value();
  assert(il2.begin() == il.begin());
  assert(il2.size() == il.size());

  return true;
}

constexpr bool test_ref() {
  { // optional(in_place_t, _Arg&&)
    Y y{1, 2};
    optional<Y&> xo(in_place, y);

    Y x2{1, 2};

    assert(*xo == x2);
    assert(&(*xo) == &y);
  }

  assert(test_ref_initializer_list());
  static_assert(test_ref_initializer_list());

  return true;
}
#endif

int main(int, char**) {
  {
    constexpr optional<int> opt(in_place, 5);
    static_assert(static_cast<bool>(opt) == true, "");
    static_assert(*opt == 5, "");

    struct test_constexpr_ctor : public optional<int> {
      constexpr test_constexpr_ctor(in_place_t, int i) : optional<int>(in_place, i) {}
    };
  }
  {
    optional<const int> opt(in_place, 5);
    assert(*opt == 5);
  }
  {
    const optional<X> opt(in_place);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == X());
  }
  {
    const optional<X> opt(in_place, 5);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == X(5));
  }
  {
    const optional<X> opt(in_place, 5, 4);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == X(5, 4));
  }
  {
    constexpr optional<Y> opt(in_place);
    static_assert(static_cast<bool>(opt) == true, "");
    static_assert(*opt == Y(), "");

    struct test_constexpr_ctor : public optional<Y> {
      constexpr test_constexpr_ctor(in_place_t) : optional<Y>(in_place) {}
    };
  }
  {
    constexpr optional<Y> opt(in_place, 5);
    static_assert(static_cast<bool>(opt) == true, "");
    static_assert(*opt == Y(5), "");

    struct test_constexpr_ctor : public optional<Y> {
      constexpr test_constexpr_ctor(in_place_t, int i) : optional<Y>(in_place, i) {}
    };
  }
  {
    constexpr optional<Y> opt(in_place, 5, 4);
    static_assert(static_cast<bool>(opt) == true, "");
    static_assert(*opt == Y(5, 4), "");

    struct test_constexpr_ctor : public optional<Y> {
      constexpr test_constexpr_ctor(in_place_t, int i, int j) : optional<Y>(in_place, i, j) {}
    };
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    try {
      const optional<Z> opt(in_place, 1);
      assert(false);
    } catch (int i) {
      assert(i == 6);
    }
  }
#endif

#if TEST_STD_VER >= 26
  test_ref();
  static_assert(test_ref());
#endif

  return 0;
}
