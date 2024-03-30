//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <vector> functions are marked [[nodiscard]]

// clang-format off

#include <vector>

void test_vector() {
  std::vector<int> vector;
  const std::vector<int> cvector;
  vector.get_allocator(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.begin();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.end();           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.begin();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.end();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.rbegin();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.rend();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.rbegin();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.rend();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.cbegin();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.cend();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.crbegin();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.crend();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.size();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.capacity();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.empty();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.max_size();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector[0];              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector[0];             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.at(0);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.at(0);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.front();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.front();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.back();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.back();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.data();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.data();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void test_vector_bool() {
  std::vector<bool> vector;
  const std::vector<bool> cvector;
  vector.get_allocator(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.begin();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.end();           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.begin();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.end();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.rbegin();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.rend();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.rbegin();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.rend();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.cbegin();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.cend();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.crbegin();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.crend();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.size();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.capacity();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.empty();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.max_size();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector[0];              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector[0];             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.at(0);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.at(0);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.front();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.front();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  vector.back();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cvector.back();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
