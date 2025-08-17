//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// template <class T, class U> constexpr bool operator==(const optional<T>& x, const U& v);
// template <class T, class U> constexpr bool operator==(const U& v, const optional<T>& x);

#include <optional>

#include "test_comparisons.h"
#include "test_macros.h"

#if TEST_STD_VER >= 26

// Test SFINAE.

static_assert(HasOperatorEqual<int, std::optional<int>>);
static_assert(HasOperatorEqual<int, std::optional<EqualityComparable>>);
static_assert(HasOperatorEqual<EqualityComparable, std::optional<EqualityComparable>>);

static_assert(!HasOperatorEqual<NonComparable, std::optional<NonComparable>>);
static_assert(!HasOperatorEqual<NonComparable, std::optional<EqualityComparable>>);

static_assert(HasOperatorEqual<std::optional<int>, int>);
static_assert(HasOperatorEqual<std::optional<EqualityComparable>, int>);
static_assert(HasOperatorEqual<std::optional<EqualityComparable>, EqualityComparable>);

static_assert(!HasOperatorEqual<std::optional<NonComparable>, NonComparable>);
static_assert(!HasOperatorEqual<std::optional<EqualityComparable>, NonComparable>);

#endif

using std::optional;

struct X {
  int i_;

  constexpr X(int i) : i_(i) {}
};

constexpr bool operator==(const X& lhs, const X& rhs) {
  return lhs.i_ == rhs.i_;
}

int main(int, char**) {
  {
    typedef X T;
    typedef optional<T> O;

    constexpr T val(2);
    constexpr O o1;      // disengaged
    constexpr O o2{1};   // engaged
    constexpr O o3{val}; // engaged

    static_assert(!(o1 == T(1)), "");
    static_assert((o2 == T(1)), "");
    static_assert(!(o3 == T(1)), "");
    static_assert((o3 == T(2)), "");
    static_assert((o3 == val), "");

    static_assert(!(T(1) == o1), "");
    static_assert((T(1) == o2), "");
    static_assert(!(T(1) == o3), "");
    static_assert((T(2) == o3), "");
    static_assert((val == o3), "");
  }
  {
    using O = optional<int>;
    constexpr O o1(42);
    static_assert(o1 == 42l, "");
    static_assert(!(101l == o1), "");
  }
  {
    using O = optional<const int>;
    constexpr O o1(42);
    static_assert(o1 == 42, "");
    static_assert(!(101 == o1), "");
  }

  return 0;
}
