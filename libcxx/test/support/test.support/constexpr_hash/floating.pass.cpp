//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// "support/constexpr_hash.h"

#include "constexpr_hash.h"

#include <cassert>
#include <type_traits>
#include <limits>
#include <cmath>

#include "test_macros.h"

#include "poisoned_hash_helper.h"

template <class T>
constexpr void test() {
  test_hash_disabled<const T, support::constexpr_hash<const T>>();
  test_hash_disabled<volatile T, support::constexpr_hash<volatile T>>();
  test_hash_disabled<const volatile T, support::constexpr_hash<const volatile T>>();

  typedef support::constexpr_hash<T> H;
  ASSERT_NOEXCEPT(H()(T()));
  H h;

  std::size_t t0   = h(0.);
  std::size_t tn0  = h(-0.);
  std::size_t tp1  = h(static_cast<T>(0.1));
  std::size_t t1   = h(1);
  std::size_t tn1  = h(-1);
  std::size_t pinf = h(INFINITY);
  std::size_t ninf = h(-INFINITY);
  assert(t0 == tn0);
  assert(t0 != tp1);
  assert(t0 != t1);
  assert(t0 != tn1);
  assert(t0 != pinf);
  assert(t0 != ninf);

  assert(tp1 != t1);
  assert(tp1 != tn1);
  assert(tp1 != pinf);
  assert(tp1 != ninf);

  assert(t1 != tn1);
  assert(t1 != pinf);
  assert(t1 != ninf);

  assert(tn1 != pinf);
  assert(tn1 != ninf);

  assert(pinf != ninf);
}

constexpr bool test_with_hash() {
  test<float>();
  test<double>();
  test<long double>();
  return true;
}

int main(int, char**) {
  assert(test_with_hash());
  static_assert(test_with_hash());
  return 0;
}
