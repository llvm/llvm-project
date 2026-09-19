//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <inplace_vector>

void test() {
  std::inplace_vector<int, 4> v;

  v.at(0);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.back();              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.begin();             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.capacity();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.cbegin();            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.cend();              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.crbegin();           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.crend();             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.data();              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.empty();             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.end();               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.front();             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.max_size();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.rbegin();            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.rend();              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.size();              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v[0];                  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.try_emplace_back(1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.try_push_back(1);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::inplace_vector<int, 0> v0;
  v0.begin();             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.capacity();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.cbegin();            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.cend();              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.crbegin();           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.crend();             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.data();              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.empty();             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.end();               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.max_size();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.rbegin();            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.rend();              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.size();              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.try_emplace_back(1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  v0.try_push_back(1);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Defined in lambdas because these are marked [[noreturn]], and would emit a 'code unreachable' warning if code followed any one of them.
  (void)[&] {
    v0.at(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  };

  (void)[&] {
    v0.front(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  };

  (void)[&] {
    v0.back(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  };

  (void)[&] {
    v0[0]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  };

  // iterator [[nodiscard]]
  {
    std::inplace_vector<int, 4>::iterator it = v.begin();
    *it;     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it[0];   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it + 1;  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    1 + it;  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - 1;  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - it; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    std::inplace_vector<int, 4>::const_iterator it = v.cbegin();
    *it;     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it[0];   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it + 1;  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    1 + it;  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - 1;  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - it; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
}
