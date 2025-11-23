//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRED: std-at-least-c++17

// <optional>

// template <class... Args>
//   constexpr explicit optional(in_place_t, Args&&... args);

#include <cassert>
#include <optional>
#include <type_traits>
#include <utility>

#include "test_macros.h"

using std::in_place;
using std::in_place_t;
using std::optional;

class X {
  int i_;
  int j_ = 0;

public:
  constexpr X() : i_(0) {}
  constexpr X(int i) : i_(i) {}
  constexpr X(int i, int j) : i_(i), j_(j) {}

  ~X() = default;

  friend constexpr bool operator==(const X& x, const X& y) { return x.i_ == y.i_ && x.j_ == y.j_; }
};

class Z {
public:
  Z(int) { TEST_THROW(6); }
};

template <typename T, typename... Args>
constexpr void test_inplace(Args... args) {
  optional<T> opt(in_place, args...);
  assert(bool(opt));
  assert(*opt == T(args...));

  struct test_constexpr_ctor : public optional<int> {
    constexpr test_constexpr_ctor(in_place_t, int i) : optional<int>(in_place, i) {}
  };
}

TEST_CONSTEXPR_CXX26 void test_throwing() {
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
}

constexpr bool test() {
  test_inplace<int>(5);
  test_inplace<const int>(5);
  test_inplace<X>();
  test_inplace<X>(5);
  test_inplace<X>(5, 4);
#if TEST_STD_VER >= 26 && 0
  test_throwing();
#endif
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  {
    test_throwing();
  }
  return 0;
}
