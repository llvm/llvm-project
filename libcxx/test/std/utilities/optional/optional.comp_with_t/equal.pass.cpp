//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17
// <optional>

// template <class T, class U> constexpr bool operator==(const optional<T>& x, const U& v);
// template <class T, class U> constexpr bool operator==(const U& v, const optional<T>& x);

#include <optional>
#include <type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

#if TEST_STD_VER >= 26
#  define STATIC_ASSERT_OPTIONAL_CMP static_assert
#else
#  define STATIC_ASSERT_OPTIONAL_CMP LIBCPP_STATIC_ASSERT
#endif

struct tester {};

template <class T, std::enable_if_t<std::is_class_v<T>, int> = 0> // intentionally underconstrained
constexpr bool operator==(const tester& a, const T& t) {
  return t.compare(a) == 0;
}
template <class T, std::enable_if_t<std::is_class_v<T>, int> = 0> // intentionally underconstrained
constexpr bool operator==(const T& t, const tester& a) {
  return t.compare(a) == 0;
}

template <class T, class U>
constexpr std::optional<bool> try_cmp_eq(const T& t, const U& u) {
  if constexpr (HasOperatorEqual<const T, const U>)
    return t == u;
  else
    return {};
}

// Test SFINAE.

STATIC_ASSERT_OPTIONAL_CMP(HasOperatorEqual<int, std::optional<int>>);
STATIC_ASSERT_OPTIONAL_CMP(HasOperatorEqual<int, std::optional<EqualityComparable>>);
STATIC_ASSERT_OPTIONAL_CMP(HasOperatorEqual<EqualityComparable, std::optional<EqualityComparable>>);

STATIC_ASSERT_OPTIONAL_CMP(!HasOperatorEqual<NonComparable, std::optional<NonComparable>>);
STATIC_ASSERT_OPTIONAL_CMP(!HasOperatorEqual<NonComparable, std::optional<EqualityComparable>>);

STATIC_ASSERT_OPTIONAL_CMP(HasOperatorEqual<std::optional<int>, int>);
STATIC_ASSERT_OPTIONAL_CMP(HasOperatorEqual<std::optional<EqualityComparable>, int>);
STATIC_ASSERT_OPTIONAL_CMP(HasOperatorEqual<std::optional<EqualityComparable>, EqualityComparable>);

STATIC_ASSERT_OPTIONAL_CMP(!HasOperatorEqual<std::optional<NonComparable>, NonComparable>);
STATIC_ASSERT_OPTIONAL_CMP(!HasOperatorEqual<std::optional<EqualityComparable>, NonComparable>);

// LWG4072: avoid ambiguity with optional's own comparison operators
STATIC_ASSERT_OPTIONAL_CMP(try_cmp_eq(std::optional<int>{}, std::optional<tester>{}) == std::optional<bool>{});
STATIC_ASSERT_OPTIONAL_CMP(try_cmp_eq(std::optional<int>{}, std::optional<int>{}) == std::optional<bool>{true});

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
