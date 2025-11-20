//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <array> functions are marked [[nodiscard]]

#include <array>
#include <utility>

#include <test_macros.h>

void array_test() {
  std::array<int, 1> a;
  const std::array<int, 1> ca{9482};

  a.begin();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.begin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.end();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.end();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.rbegin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.rbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.rend();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.rend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.cbegin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.cbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.cend();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.cend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.crbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.crbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.crend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.crend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  a.size();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.empty();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  a[0];     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca[0];    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.at(0);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.at(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  a.front();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.front(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.back();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.back();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  a.data();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.data(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::get<0>(a);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(ca); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(std::move(a));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(std::move(ca));

#if TEST_STD_VER >= 20
  std::to_array("zmt");    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::to_array({94, 82}); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}

void empty_array_test() {
  std::array<int, 0> a;
  const std::array<int, 0> ca;

  a.begin();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.begin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.end();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.end();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.rbegin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.rbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.rend();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.rend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.cbegin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.cbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.cend();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.cend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.crbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.crbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.crend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.crend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  a.size();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.empty();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  a[0];     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca[0];    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.at(0);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.at(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  a.front();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.front(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.back();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.back();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  a.data();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.data(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
