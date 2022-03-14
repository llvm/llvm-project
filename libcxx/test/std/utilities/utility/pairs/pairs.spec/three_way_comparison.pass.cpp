//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template <class T1, class T2> bool operator<=>(const pair<T1,T2>&, const pair<T1,T2>&);

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cassert>
#include <compare>
#include <limits>
#include <type_traits> // std::is_constant_evaluated
#include <utility>

#include "test_macros.h"

template<class T> concept HasEqual = requires(T t) { t == t; };
template<class T> concept HasLess = requires(T t) { t < t; };
template<class T> concept HasSpaceship = requires(T t) { t <=> t; };

constexpr bool test() {
  {
    // Pairs of types that both have strong ordering should compare with strong ordering.
    using P = std::pair<int, int>;
    ASSERT_SAME_TYPE(decltype(P() <=> P()), std::strong_ordering);
    assert((P(1, 1) <=> P(1, 2)) == std::strong_ordering::less);
    assert((P(2, 1) <=> P(1, 2)) == std::strong_ordering::greater);
    assert((P(0, 0) <=> P(0, 0)) == std::strong_ordering::equal);
  }
  {
    // Pairs of int and a type with no spaceship operator should compare with weak ordering.
    struct NoSpaceship {
      int value;
      constexpr bool operator==(const NoSpaceship&) const = default;
      constexpr bool operator<(const NoSpaceship& other) const { return value < other.value; }
    };
    using P = std::pair<int, NoSpaceship>;
    ASSERT_SAME_TYPE(decltype(P() <=> P()), std::weak_ordering);
    assert((P(1, {1}) <=> P(1, {2})) == std::weak_ordering::less);
    assert((P(2, {1}) <=> P(1, {2})) == std::weak_ordering::greater);
    assert((P(0, {0}) <=> P(0, {0})) == std::weak_ordering::equivalent);
  }
  {
    // Pairs of int (strongly ordered) and double (partially ordered) should compare with partial ordering.
    using P = std::pair<int, double>;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    ASSERT_SAME_TYPE(decltype(P() <=> P()), std::partial_ordering);
    assert((P(1, 1.0) <=> P(1, 2.0)) == std::partial_ordering::less);
    assert((P(1, 1.0) <=> P(1, 1.0)) == std::partial_ordering::equivalent);
    assert((P(1, -0.0) <=> P(1, 0.0)) == std::partial_ordering::equivalent);
    assert((P(1, 2.0) <=> P(1, 1.0)) == std::partial_ordering::greater);
    assert((P(1, nan) <=> P(2, nan)) == std::partial_ordering::less);
    assert((P(2, nan) <=> P(1, nan)) == std::partial_ordering::greater);
    assert((P(1, nan) <=> P(1, nan)) == std::partial_ordering::unordered);
  }
  {
    using P = std::pair<double, int>;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    ASSERT_SAME_TYPE(decltype(P() <=> P()), std::partial_ordering);
    assert((P(2.0, 1) <=> P(1.0, 2)) == std::partial_ordering::greater);
    assert((P(1.0, 1) <=> P(1.0, 2)) == std::partial_ordering::less);
    assert((P(nan, 1) <=> P(nan, 2)) == std::partial_ordering::unordered);
  }
  {
    struct NoRelative {
      constexpr bool operator==(const NoRelative&) const;
    };
    static_assert(HasEqual<std::pair<int, NoRelative>>);
    static_assert(!HasLess<std::pair<int, NoRelative>>);
    static_assert(!HasSpaceship<std::pair<int, NoRelative>>);
  }
  {
    struct NoLessThan {
      constexpr bool operator==(const NoLessThan&) const;
      constexpr bool operator>(const NoLessThan&) const;
    };
    static_assert(HasEqual<std::pair<int, NoLessThan>>);
    static_assert(!HasLess<std::pair<int, NoLessThan>>);
    static_assert(!HasSpaceship<std::pair<int, NoLessThan>>);
  }

#ifdef TEST_COMPILER_GCC
  // GCC cannot evaluate NaN @ non-NaN constexpr, so test that runtime-only.
  if (!std::is_constant_evaluated())
#endif
  {
    {
      using P = std::pair<int, double>;
      constexpr double nan = std::numeric_limits<double>::quiet_NaN();
      assert((P(1, 2.0) <=> P(1, nan)) == std::partial_ordering::unordered);
    }
    {
      using P = std::pair<double, int>;
      constexpr double nan = std::numeric_limits<double>::quiet_NaN();
      assert((P(1.0, 1) <=> P(nan, 2)) == std::partial_ordering::unordered);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
