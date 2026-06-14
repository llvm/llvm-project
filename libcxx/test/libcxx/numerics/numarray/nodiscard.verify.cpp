//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// Check that functions are marked [[nodiscard]]

#include <valarray>
#include <utility>

#include "test_macros.h"

int func(int);

int cfunc(const int&);

void test() {
  std::slice s;

  s.size();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  s.start();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  s.stride(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::gslice gs;

  std::valarray<int> va;
  const std::valarray<int> cva;

  std::valarray<bool> bva;
  std::valarray<std::size_t> sva;

  cva[0];   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  va[0];    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cva[s];   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  va[s];    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cva[gs];  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  va[gs];   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cva[bva]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  va[bva];  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 11
  cva[std::move(bva)]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cva[std::move(bva)]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
  cva[sva]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  va[sva];  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 11
  cva[std::move(sva)]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cva[std::move(sva)]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  va.size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  va.sum(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  va.min(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  va.max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  va.shift(1);     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  va.cshift(1);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  va.apply(func);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  va.apply(cfunc); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  gs.size();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  gs.start();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  gs.stride(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::abs(va);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::acos(va);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::asin(va);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::atan(va);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::atan2(va, va); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::atan2(va, 94); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::atan2(82, va); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cos(va);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cosh(va);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::exp(va);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::log(va);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::log10(va);     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::pow(va, va);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::pow(va, 94);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::pow(82, va);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::sin(va);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::sinh(va);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::sqrt(va);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::tan(va);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::tanh(va);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::begin(va);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::begin(cva); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::end(va);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::end(cva);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
