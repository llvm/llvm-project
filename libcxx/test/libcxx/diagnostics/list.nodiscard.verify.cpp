//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that <list> functions are marked [[nodiscard]]

#include <list>

void test() {
  std::list<int> l;
  const std::list<int> cl;

  l.get_allocator(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  l.size();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  l.empty();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  l.max_size();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  l.begin();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cl.begin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  l.end();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cl.end();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  l.cbegin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cl.cbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  l.cend();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cl.cend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  l.rbegin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cl.rbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  l.rend();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cl.rend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  l.crbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cl.crbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  l.crend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cl.crend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  l.front();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cl.front(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  l.back();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cl.back();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
