//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// template <class T, class U> constexpr bool operator<(const optional<T>& x, const U& v);
// template <class T, class U> constexpr bool operator<(const U& v, const optional<T>& x);

#include <optional>

#include "test_comparisons.h"
#include "test_macros.h"

#if TEST_STD_VER >= 26

// Test SFINAE.
static_assert(HasOperatorLessThan<std::optional<ThreeWayComparable>, int>);
static_assert(HasOperatorLessThan<std::optional<ThreeWayComparable>, ThreeWayComparable>);

static_assert(!HasOperatorLessThan<std::optional<NonComparable>, NonComparable>);
static_assert(!HasOperatorLessThan<std::optional<ThreeWayComparable>, NonComparable>);
static_assert(!HasOperatorLessThan<std::optional<NonComparable>, ThreeWayComparable>);

static_assert(HasOperatorLessThan<int, std::optional<ThreeWayComparable>>);
static_assert(HasOperatorLessThan<ThreeWayComparable, std::optional<ThreeWayComparable>>);

static_assert(!HasOperatorLessThan<NonComparable, std::optional<NonComparable>>);
static_assert(!HasOperatorLessThan<NonComparable, std::optional<ThreeWayComparable>>);
static_assert(!HasOperatorLessThan<ThreeWayComparable, std::optional<NonComparable>>);

#endif

using std::optional;

struct X {
  int i_;

  constexpr X(int i) : i_(i) {}
};

constexpr bool operator<(const X& lhs, const X& rhs) { return lhs.i_ < rhs.i_; }

int main(int, char**) {
  {
    typedef X T;
    typedef optional<T> O;

    constexpr T val(2);
    constexpr O o1;      // disengaged
    constexpr O o2{1};   // engaged
    constexpr O o3{val}; // engaged

    static_assert((o1 < T(1)), "");
    static_assert(!(o2 < T(1)), ""); // equal
    static_assert(!(o3 < T(1)), "");
    static_assert((o2 < val), "");
    static_assert(!(o3 < val), ""); // equal
    static_assert((o3 < T(3)), "");

    static_assert(!(T(1) < o1), "");
    static_assert(!(T(1) < o2), ""); // equal
    static_assert((T(1) < o3), "");
    static_assert(!(val < o2), "");
    static_assert(!(val < o3), ""); // equal
    static_assert(!(T(3) < o3), "");
  }
  {
    using O = optional<int>;
    constexpr O o1(42);
    static_assert(o1 < 101l, "");
    static_assert(!(42l < o1), "");
  }
  {
    using O = optional<const int>;
    constexpr O o1(42);
    static_assert(o1 < 101, "");
    static_assert(!(42 < o1), "");
  }

  return 0;
}
