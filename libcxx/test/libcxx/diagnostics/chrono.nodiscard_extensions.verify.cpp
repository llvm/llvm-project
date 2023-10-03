//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that format functions are marked [[nodiscard]] as a conforming extension

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-incomplete-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

#include <chrono>

#include "test_macros.h"

void test() {
  std::chrono::tzdb_list& list = std::chrono::get_tzdb_list();
  list.front();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  list.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  list.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  list.cbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  list.cend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  namespace crno = std::chrono;
  crno::get_tzdb_list();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  crno::get_tzdb();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  crno::remote_version(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
