//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// Test that during constant evaluation, std::(ranges::}uninitialized_default_construct(_n)
// do not initialize elements to determined values.

#include <memory>

constexpr int test_uninitialized_default_construct() {
  int c[2];
  std::uninitialized_default_construct(&c[0], &c[2]);
  return c[0]; // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}

// expected-error@+1 {{static assertion expression is not an integral constant expression}}
static_assert(test_uninitialized_default_construct() == 0);

constexpr int test_uninitialized_default_construct_n() {
  int c[2];
  std::uninitialized_default_construct_n(&c[0], 2);
  return c[0]; // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}

// expected-error@+1 {{static assertion expression is not an integral constant expression}}
static_assert(test_uninitialized_default_construct_n() == 0);

constexpr int test_ranges_uninitialized_default_construct_iter_sent() {
  int c[2];
  std::ranges::uninitialized_default_construct(&c[0], &c[2]);
  return c[0]; // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}

// expected-error@+1 {{static assertion expression is not an integral constant expression}}
static_assert(test_ranges_uninitialized_default_construct_iter_sent() == 0);

constexpr int test_ranges_uninitialized_default_construct_range() {
  int c[2];
  std::ranges::uninitialized_default_construct(c);
  return c[0]; // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}

// expected-error@+1 {{static assertion expression is not an integral constant expression}}
static_assert(test_ranges_uninitialized_default_construct_range() == 0);

constexpr int test_ranges_uninitialized_default_construct_n() {
  int c[2];
  std::ranges::uninitialized_default_construct_n(&c[0], 2);
  return c[0]; // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}

// expected-error@+1 {{static assertion expression is not an integral constant expression}}
static_assert(test_ranges_uninitialized_default_construct_n() == 0);
