//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that <forward_list> functions are marked [[nodiscard]]

#include <forward_list>

void test() {
  std::forward_list<int> fl;
  const std::forward_list<int> cfl;

  fl.get_allocator(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fl.begin();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfl.begin();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fl.end();            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfl.end();           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fl.cbegin();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfl.cbegin();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fl.cend();           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfl.cend();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fl.before_begin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfl.before_begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fl.cbefore_begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfl.cbefore_begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fl.empty();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fl.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fl.front();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfl.front(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
