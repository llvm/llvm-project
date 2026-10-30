//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17
// <optional>

// template <class T, class U> constexpr bool operator>(const optional<T>& x, const U& v);
// template <class T, class U> constexpr bool operator>(const U& v, const optional<T>& x);

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
constexpr bool operator>(const tester& a, const T& t) {
  return t.compare(a) < 0;
}
template <class T, std::enable_if_t<std::is_class_v<T>, int> = 0> // intentionally underconstrained
constexpr bool operator>(const T& t, const tester& a) {
  return t.compare(a) > 0;
}

template <class T, class U>
constexpr std::optional<bool> try_cmp_gt(const T& t, const U& u) {
  if constexpr (HasOperatorGreaterThan<const T, const U>)
    return t > u;
  else
    return {};
}

// Test SFINAE.
STATIC_ASSERT_OPTIONAL_CMP(HasOperatorGreaterThan<std::optional<TotallyOrdered>, int>);
STATIC_ASSERT_OPTIONAL_CMP(HasOperatorGreaterThan<std::optional<TotallyOrdered>, TotallyOrdered>);

STATIC_ASSERT_OPTIONAL_CMP(!HasOperatorGreaterThan<std::optional<NonComparable>, NonComparable>);
STATIC_ASSERT_OPTIONAL_CMP(!HasOperatorGreaterThan<std::optional<TotallyOrdered>, NonComparable>);
STATIC_ASSERT_OPTIONAL_CMP(!HasOperatorGreaterThan<std::optional<NonComparable>, TotallyOrdered>);

STATIC_ASSERT_OPTIONAL_CMP(HasOperatorGreaterThan<int, std::optional<TotallyOrdered>>);
STATIC_ASSERT_OPTIONAL_CMP(HasOperatorGreaterThan<TotallyOrdered, std::optional<TotallyOrdered>>);

STATIC_ASSERT_OPTIONAL_CMP(!HasOperatorGreaterThan<NonComparable, std::optional<NonComparable>>);
STATIC_ASSERT_OPTIONAL_CMP(!HasOperatorGreaterThan<NonComparable, std::optional<TotallyOrdered>>);
STATIC_ASSERT_OPTIONAL_CMP(!HasOperatorGreaterThan<TotallyOrdered, std::optional<NonComparable>>);

// LWG4072: avoid ambiguity with optional's own comparison operators
STATIC_ASSERT_OPTIONAL_CMP(try_cmp_gt(std::optional<int>{}, std::optional<tester>{}) == std::optional<bool>{});
STATIC_ASSERT_OPTIONAL_CMP(try_cmp_gt(std::optional<int>{}, std::optional<int>{}) == std::optional<bool>{false});

using std::optional;

struct X {
  int i_;

  constexpr X(int i) : i_(i) {}
};

constexpr bool operator>(const X& lhs, const X& rhs) { return lhs.i_ > rhs.i_; }

int main(int, char**) {
  {
    typedef X T;
    typedef optional<T> O;

    constexpr T val(2);
    constexpr O o1;      // disengaged
    constexpr O o2{1};   // engaged
    constexpr O o3{val}; // engaged

    static_assert(!(o1 > T(1)), "");
    static_assert(!(o2 > T(1)), ""); // equal
    static_assert((o3 > T(1)), "");
    static_assert(!(o2 > val), "");
    static_assert(!(o3 > val), ""); // equal
    static_assert(!(o3 > T(3)), "");

    static_assert((T(1) > o1), "");
    static_assert(!(T(1) > o2), ""); // equal
    static_assert(!(T(1) > o3), "");
    static_assert((val > o2), "");
    static_assert(!(val > o3), ""); // equal
    static_assert((T(3) > o3), "");
  }
  {
    using O = optional<int>;
    constexpr O o1(42);
    static_assert(o1 > 11l, "");
    static_assert(!(42l > o1), "");
  }
  {
    using O = optional<const int>;
    constexpr O o1(42);
    static_assert(o1 > 11, "");
    static_assert(!(42 > o1), "");
  }

  return 0;
}
