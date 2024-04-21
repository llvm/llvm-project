//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <functional>

// class reference_wrapper

// // [refwrap.comparisons], comparisons

// friend constexpr synth-three-way-result<T> operator<=>(reference_wrapper, reference_wrapper);          // Since C++26
// friend constexpr synth-three-way-result<T> operator<=>(reference_wrapper, const T&);                   // Since C++26
// friend constexpr synth-three-way-result<T> operator<=>(reference_wrapper, reference_wrapper<const T>); // Since C++26

#include <cassert>
#include <concepts>
#include <functional>

#include "test_comparisons.h"
#include "test_macros.h"

struct NonComparable {};

static_assert(!std::three_way_comparable<NonComparable>);

// Test SFINAE.

template <class _Tp>
concept BooleanTestableImpl = std::convertible_to<_Tp, bool>;

template <class _Tp>
concept BooleanTestable = BooleanTestableImpl<_Tp> && requires(_Tp&& __t) {
  { !std::forward<_Tp>(__t) } -> BooleanTestableImpl;
};

template <typename T>
concept HasEqualityOperatorWithInt = requires(T t, int i) {
  { t < i } -> BooleanTestable;
  { i < t } -> BooleanTestable;
};

// refwrap, refwrap
static_assert(std::three_way_comparable<std::reference_wrapper<StrongOrder>>);
static_assert(std::three_way_comparable<std::reference_wrapper<WeakOrder>>);
static_assert(std::three_way_comparable<std::reference_wrapper<PartialOrder>>);
// refwrap, const&
static_assert(HasEqualityOperatorWithInt<std::reference_wrapper<StrongOrder>>);
static_assert(HasEqualityOperatorWithInt<std::reference_wrapper<WeakOrder>>);
static_assert(HasEqualityOperatorWithInt<std::reference_wrapper<PartialOrder>>);
// refwrap, refwrap<const>
static_assert(std::three_way_comparable_with<std::reference_wrapper<StrongOrder>, const StrongOrder>);
static_assert(std::three_way_comparable_with<std::reference_wrapper<WeakOrder>, const WeakOrder>);
static_assert(std::three_way_comparable_with<std::reference_wrapper<PartialOrder>, const PartialOrder>);

// refwrap, refwrap
static_assert(!std::three_way_comparable<std::reference_wrapper<NonComparable>>);
// refwrap, const&
static_assert(!HasEqualityOperatorWithInt<std::reference_wrapper<NonComparable>>);
// refwrap, refwrap<const>
static_assert(!std::three_way_comparable_with<std::reference_wrapper<StrongOrder>, const NonComparable>);
static_assert(!std::three_way_comparable_with<std::reference_wrapper<WeakOrder>, const NonComparable>);
static_assert(!std::three_way_comparable_with<std::reference_wrapper<PartialOrder>, const NonComparable>);

// Test comparisons.

template <typename T, typename Order>
constexpr void test() {
  T t{47};

  T bigger{94};
  T smaller{82};

  T unordered{std::numeric_limits<int>::min()};

  // operator<=>(reference_wrapper, reference_wrapper)
  {
    // Identical contents
    {
      std::reference_wrapper<T> rw1{t};
      std::reference_wrapper<T> rw2{t};
      assert(testOrder(rw1, rw2, Order::equivalent));
    }
    // Less
    {
      std::reference_wrapper<T> rw1{smaller};
      std::reference_wrapper<T> rw2{bigger};
      assert(testOrder(rw1, rw2, Order::less));
    }
    // Greater
    {
      std::reference_wrapper<T> rw1{bigger};
      std::reference_wrapper<T> rw2{smaller};
      assert(testOrder(rw1, rw2, Order::greater));
    }
    // Unordered
    if constexpr (std::same_as<T, PartialOrder>) {
      std::reference_wrapper<T> rw1{bigger};
      std::reference_wrapper<T> rw2{unordered};
      assert(testOrder(rw1, rw2, Order::unordered));
    }
  }

  // operator<=>(reference_wrapper, const T&)
  {
    // Identical contents
    {
      std::reference_wrapper<T> rw1{t};
      assert(testOrder(rw1, t, Order::equivalent));
    }
    // Less
    {
      std::reference_wrapper<T> rw1{smaller};
      assert(testOrder(rw1, bigger, Order::less));
    }
    // Greater
    {
      std::reference_wrapper<T> rw1{bigger};
      assert(testOrder(rw1, smaller, Order::greater));
    }
    // Unordered
    if constexpr (std::same_as<T, PartialOrder>) {
      std::reference_wrapper<T> rw1{bigger};
      assert(testOrder(rw1, unordered, Order::unordered));
    }
  }

  // operator<=>(reference_wrapper, reference_wrapper<const T>)
  {
    // Identical contents
    {
      std::reference_wrapper<T> rw1{t};
      std::reference_wrapper<const T> rw2{t};
      assert(testOrder(rw1, rw2, Order::equivalent));
    }
    // Less
    {
      std::reference_wrapper<T> rw1{smaller};
      std::reference_wrapper<const T> rw2{bigger};
      assert(testOrder(rw1, rw2, Order::less));
    }
    // Greater
    {
      std::reference_wrapper<T> rw1{bigger};
      std::reference_wrapper<const T> rw2{smaller};
      assert(testOrder(rw1, rw2, Order::greater));
    }
    // Unordered
    if constexpr (std::same_as<T, PartialOrder>) {
      std::reference_wrapper<T> rw1{bigger};
      std::reference_wrapper<const T> rw2{unordered};
      assert(testOrder(rw1, rw2, Order::unordered));
    }
  }
}

constexpr bool test() {
  test<int, std::strong_ordering>();
  test<StrongOrder, std::strong_ordering>();
  test<int, std::weak_ordering>();
  test<WeakOrder, std::weak_ordering>();
  test<int, std::partial_ordering>();
  test<PartialOrder, std::partial_ordering>();

  // `LessAndEqComp` does not have `operator<=>`. Ordering is synthesized based on `operator<`
  test<LessAndEqComp, std::weak_ordering>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
