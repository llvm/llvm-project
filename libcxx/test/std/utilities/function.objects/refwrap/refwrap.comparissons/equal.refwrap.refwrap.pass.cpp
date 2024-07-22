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

// [refwrap.comparisons], comparisons
// friend constexpr bool operator==(reference_wrapper, reference_wrapper);                                // Since C++26

#include <cassert>
#include <concepts>
#include <functional>

#include "test_comparisons.h"
#include "test_macros.h"

#include "helper_concepts.h"
#include "helper_types.h"

// Test SFINAE.

static_assert(std::equality_comparable<std::reference_wrapper<EqualityComparable>>);

static_assert(!std::equality_comparable<std::reference_wrapper<NonComparable>>);

// Test equality.

template <typename T>
constexpr void test() {
  T i{92};
  T j{84};

  std::reference_wrapper<T> rw1{i};
  std::reference_wrapper<T> rw2 = rw1;
  std::reference_wrapper<T> rw3{j};
  std::reference_wrapper<const T> crw1{i};
  std::reference_wrapper<const T> crw3{j};

  AssertEqualityReturnBool<decltype(rw1), decltype(rw2)>();
  assert(testEquality(rw1, rw2, true));
  assert(testEquality(rw1, rw3, false));
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
