//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <span>

// Check that functions are marked [[nodiscard]]

#include <span>

#include "test_macros.h"

void test() {
  { // Test with a static extent
    std::span<int, 0> sp;

    sp.first<0>();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.last<0>();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.first(0);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.last(0);         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.subspan<0, 0>(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.subspan(0, 0);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.size();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.size_bytes();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.empty();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp[0];              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 26
    sp.at(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
    sp.front();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.back();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.data();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.rbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.rend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::as_bytes(sp); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_writable_bytes(sp);
  }
  { // Test with a dynamic extent
    std::span<int> sp;

    sp.first<0>();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.last<0>();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.first(0);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.last(0);         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.subspan<0, 0>(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.subspan(0, 0);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.size();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.size_bytes();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.empty();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp[0];              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 26
    sp.at(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
    sp.front();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.back();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.data();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.rbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sp.rend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::as_bytes(sp); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_writable_bytes(sp);
  }
}
