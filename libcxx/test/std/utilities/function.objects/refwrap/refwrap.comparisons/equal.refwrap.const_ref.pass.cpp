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

// friend constexpr bool operator==(reference_wrapper, const T&);                                         // Since C++26

#include <cassert>
#include <concepts>
#include <functional>

#include "test_comparisons.h"
#include "test_macros.h"

// Test SFINAE.

static_assert(HasOperatorEqual<std::reference_wrapper<EqualityComparable>>);
static_assert(HasOperatorEqual<std::reference_wrapper<EqualityComparable>, int>);

static_assert(!HasOperatorEqual<std::reference_wrapper<NonComparable>>);
static_assert(!HasOperatorEqual<std::reference_wrapper<NonComparable>, int>);

// Test equality.

template <typename T>
constexpr void test() {
  T i{92};
  T j{84};

  std::reference_wrapper<T> rw1{i};

  // refwrap, const&
  AssertEqualityReturnBool<decltype(rw1), decltype(i)>();
  assert(testEquality(rw1, i, true));
  assert(testEquality(rw1, j, false));
}

constexpr bool test() {
  test<int>();
  test<EqualityComparable>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
