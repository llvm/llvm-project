//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17
// <optional>

// template <class T, class U> constexpr bool operator!=(const optional<T>& x, const optional<U>& y);

#include <cassert>
#include <optional>
#include <type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

#if TEST_STD_VER >= 26
#  define STATIC_ASSERT_OPTIONAL_CMP static_assert
#else
#  define STATIC_ASSERT_OPTIONAL_CMP LIBCPP_STATIC_ASSERT
#endif

// Test SFINAE.

STATIC_ASSERT_OPTIONAL_CMP(HasOperatorNotEqual<std::optional<int>>);
STATIC_ASSERT_OPTIONAL_CMP(HasOperatorNotEqual<std::optional<EqualityComparable>>);
STATIC_ASSERT_OPTIONAL_CMP(HasOperatorNotEqual<std::optional<EqualityComparable>, std::optional<int>>);

STATIC_ASSERT_OPTIONAL_CMP(!HasOperatorNotEqual<std::optional<NonComparable>>);
STATIC_ASSERT_OPTIONAL_CMP(!HasOperatorNotEqual<std::optional<EqualityComparable>, std::optional<NonComparable>>);

using std::optional;

struct X {
  int i_;

  constexpr X(int i) : i_(i) {}
};

constexpr bool operator!=(const X& lhs, const X& rhs) {
  return lhs.i_ != rhs.i_;
}

int main(int, char**) {
  {
    typedef X T;
    typedef optional<T> O;

    constexpr O o1;    // disengaged
    constexpr O o2;    // disengaged
    constexpr O o3{1}; // engaged
    constexpr O o4{2}; // engaged
    constexpr O o5{1}; // engaged

    static_assert(!(o1 != o1), "");
    static_assert(!(o1 != o2), "");
    static_assert((o1 != o3), "");
    static_assert((o1 != o4), "");
    static_assert((o1 != o5), "");

    static_assert(!(o2 != o1), "");
    static_assert(!(o2 != o2), "");
    static_assert((o2 != o3), "");
    static_assert((o2 != o4), "");
    static_assert((o2 != o5), "");

    static_assert((o3 != o1), "");
    static_assert((o3 != o2), "");
    static_assert(!(o3 != o3), "");
    static_assert((o3 != o4), "");
    static_assert(!(o3 != o5), "");

    static_assert((o4 != o1), "");
    static_assert((o4 != o2), "");
    static_assert((o4 != o3), "");
    static_assert(!(o4 != o4), "");
    static_assert((o4 != o5), "");

    static_assert((o5 != o1), "");
    static_assert((o5 != o2), "");
    static_assert(!(o5 != o3), "");
    static_assert((o5 != o4), "");
    static_assert(!(o5 != o5), "");
  }
  {
    using O1 = optional<int>;
    using O2 = optional<long>;
    constexpr O1 o1(42);
    static_assert(o1 != O2(101), "");
    static_assert(!(O2(42) != o1), "");
  }
  {
    using O1 = optional<int>;
    using O2 = optional<const int>;
    constexpr O1 o1(42);
    static_assert(o1 != O2(101), "");
    static_assert(!(O2(42) != o1), "");
  }

  return 0;
}
