//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast

// <valarray>

// Test that the operands of a binary operation on valarrays must have the same size.

#include <valarray>

#include "check_assertion.h"

int main(int, char**) {
  std::valarray<int> a(1, 3);
  std::valarray<int> b(2, 3);
  std::valarray<int> e(3, 2);

  (void)(a + b); // Check that there's no assertion for operands of the same size.
  (void)(a + b + a);

  // valarray and valarray
  TEST_LIBCPP_ASSERT_FAILURE(a + e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(e + a, "valarray binary operation size mismatch");

  // expression and valarray
  TEST_LIBCPP_ASSERT_FAILURE(a + b + e, "valarray binary operation size mismatch");

  // valarray and expression
  TEST_LIBCPP_ASSERT_FAILURE(e + (a + b), "valarray binary operation size mismatch");

  // expression and expression
  TEST_LIBCPP_ASSERT_FAILURE((a + b) + (e + e), "valarray binary operation size mismatch");

  TEST_LIBCPP_ASSERT_FAILURE(a * e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a / e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a % e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a - e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a ^ e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a & e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a | e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a << e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a >> e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a && e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a || e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a == e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a != e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a < e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a > e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a <= e, "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(a >= e, "valarray binary operation size mismatch");

  std::valarray<double> x(1.0, 3);
  std::valarray<double> y(2.0, 2);
  TEST_LIBCPP_ASSERT_FAILURE(std::atan2(x, y), "valarray binary operation size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(std::pow(x, y), "valarray binary operation size mismatch");

  return 0;
}
