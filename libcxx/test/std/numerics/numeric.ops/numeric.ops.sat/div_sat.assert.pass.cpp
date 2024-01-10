//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: availability-verbose_abort-missing

// <numeric>

// template<class T>
// constexpr T div_sat(T x, T y) noexcept;                     // freestanding

#include <cassert>
#include <concepts>
#include <limits>
#include <numeric>

#include "check_assertion.h"

template <typename IntegerT>
constexpr void test() {
  TEST_LIBCPP_ASSERT_FAILURE((void)std::div_sat(IntegerT{3}, IntegerT{0}), "Division by 0 is undefined");
}

constexpr bool test() {
  // signed
  test<signed char>();
  test<short int>();
  test<int>();
  test<long int>();
  test<long long int>();
  // unsigned
  test<unsigned char>();
  test<unsigned short int>();
  test<unsigned int>();
  test<unsigned long int>();
  test<unsigned long long int>();

  return true;
}

int main(int, char**) {
  assert(test());

  return 0;
}
