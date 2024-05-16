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
#include <numeric>

#include "check_assertion.h"
#include "test_macros.h"

template <typename IntegerT>
void test_runtime_assertion() {
  TEST_LIBCPP_ASSERT_FAILURE((void)std::div_sat(IntegerT{27}, IntegerT{0}), "Division by 0 is undefined");
}

bool test() {
  // Signed
  test_runtime_assertion<signed char>();
  test_runtime_assertion<short int>();
  test_runtime_assertion<int>();
  test_runtime_assertion<long int>();
  test_runtime_assertion<long long int>();
#ifndef TEST_HAS_NO_INT128
  test_runtime_assertion<__int128_t>();
#endif
  // Unsigned
  test_runtime_assertion<unsigned char>();
  test_runtime_assertion<unsigned short int>();
  test_runtime_assertion<unsigned int>();
  test_runtime_assertion<unsigned long int>();
  test_runtime_assertion<unsigned long long int>();
#ifndef TEST_HAS_NO_INT128
  test_runtime_assertion<__uint128_t>();
#endif

  return true;
}

int main(int, char**) {
  test();

  return 0;
}
