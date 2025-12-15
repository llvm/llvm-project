//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that <deque> functions are marked [[nodiscard]]

#include <deque>

void test() {
  std::deque<int> d;
  const std::deque<int> cd;

  d.begin();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cd.begin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  d.end();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cd.end();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  d.rbegin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cd.rbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  d.rend();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cd.rend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cd.cbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cd.cend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cd.crbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cd.crend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  d.size();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  d.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  d.empty();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  d[0];       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cd[0];      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  d.at(0);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cd.at(0);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  d.front();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cd.front(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  d.back();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cd.back();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
