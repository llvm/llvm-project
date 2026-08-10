//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

#include <stacktrace>

void test() {
  std::stacktrace st = std::stacktrace::current();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.get_allocator();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.cbegin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.cend();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.rbegin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.rend();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.crbegin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.crend();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.empty();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.max_size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st[0];
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.at(0);
}
