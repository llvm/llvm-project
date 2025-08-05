//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <functional>

// class reference_wrapper

// [refwrap.comparisons], comparisons

// friend constexpr auto operator<=>(reference_wrapper, const T&);                   // Since C++26

#include <cassert>
#include <concepts>
#include <functional>

#include "test_comparisons.h"
#include "test_macros.h"

// Test SFINAE.

static_assert(HasOperatorSpaceship<std::reference_wrapper<StrongOrder>, int>);
static_assert(HasOperatorSpaceship<std::reference_wrapper<WeakOrder>, int>);
static_assert(HasOperatorSpaceship<std::reference_wrapper<PartialOrder>, int>);

static_assert(!HasOperatorSpaceship<std::reference_wrapper<NonComparable>, int>);

// Test comparisons.

template <typename T, typename Order>
constexpr void test() {
  T t{47};

  T bigger{94};
  T smaller{82};

  T unordered{std::numeric_limits<int>::min()};

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
