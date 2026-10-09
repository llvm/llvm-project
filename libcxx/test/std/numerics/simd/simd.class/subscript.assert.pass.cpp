//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <simd>

// Test hardening assertions for std::datapar::simd.

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// REQUIRES: std-at-least-c++26

#include <simd>

#include "check_assertion.h"
#include "../utils.h"

namespace dp = std::datapar;

int main(int, char**) {
  simd_utils::test_sizes([]<int N>(std::integral_constant<int, N>) {
    dp::simd<int, N> vec;
    TEST_LIBCPP_ASSERT_FAILURE(vec[-1], "simd::operator[] out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(vec[N], "simd::operator[] out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(vec[N + 1], "simd::operator[] out of bounds");
  });
  return 0;
}
