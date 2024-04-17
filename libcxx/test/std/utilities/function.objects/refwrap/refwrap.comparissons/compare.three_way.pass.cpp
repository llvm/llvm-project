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
#include <functional>

#include "test_comparisons.h"
#include "test_macros.h"

template <typename Order>
constexpr void test() {
  int integer = 47;

  int bigger  = 94;
  int smaller = 82;

  // Identical contents
  {
    std::reference_wrapper<int> rw1{integer};
    std::reference_wrapper<int> rw2{integer};
    assert(testOrder(rw1, rw2, Order::equivalent));
  }
  // Less
  {
    std::reference_wrapper<int> rw1{smaller};
    std::reference_wrapper<int> rw2{bigger};
    assert(testOrder(rw1, rw2, Order::less));
  }
  // Greater
  {
    std::reference_wrapper<int> rw1{bigger};
    std::reference_wrapper<int> rw2{smaller};
    assert(testOrder(rw1, rw2, Order::greater));
  }
}

constexpr bool test() {
  test<std::strong_ordering>();
  test<std::weak_ordering>();
  test<std::partial_ordering>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
