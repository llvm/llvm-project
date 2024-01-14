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

#define ASSERT_CONSTEXPR(Expr) static_assert(__builtin_constant_p(Expr))
#define ASSERT_NOT_CONSTEXPR(Expr) static_assert(!__builtin_constant_p(Expr));

template <typename IntegerT>
void test_runtime_assertion() {
  TEST_LIBCPP_ASSERT_FAILURE((void)std::div_sat(IntegerT{3}, IntegerT{0}), "Division by 0 is undefined");
}

template <typename IntegerT>
void test_constexpr() {
  ASSERT_CONSTEXPR(std::div_sat(IntegerT{90}, IntegerT{84}));
  ASSERT_NOT_CONSTEXPR(std::div_sat(IntegerT{90}, IntegerT{0}));
}

bool test() {
  // Signed
  test_runtime_assertion<signed char>();
  test_runtime_assertion<short int>();
  test_runtime_assertion<int>();
  test_runtime_assertion<long int>();
  test_runtime_assertion<long long int>();
  // Unsigned
  test_runtime_assertion<unsigned char>();
  test_runtime_assertion<unsigned short int>();
  test_runtime_assertion<unsigned int>();
  test_runtime_assertion<unsigned long int>();
  test_runtime_assertion<unsigned long long int>();

  // Signed
  test_constexpr<signed char>();
  test_constexpr<short int>();
  test_constexpr<int>();
  test_constexpr<long int>();
  test_constexpr<long long int>();
  // Unsigned
  test_constexpr<unsigned char>();
  test_constexpr<unsigned short int>();
  test_constexpr<unsigned int>();
  test_constexpr<unsigned long int>();
  test_constexpr<unsigned long long int>();

  return true;
}

int main(int, char**) {
  assert(test());

  return 0;
}
