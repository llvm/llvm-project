//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-experimental-syncstream

// <synchstream>

// Check that functions are marked [[nodiscard]]

#include <sstream>
#include <syncstream>

#include "test_macros.h"

void test() {
  {
    std::basic_syncbuf<char> sb;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sb.get_wrapped();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sb.get_allocator();
  }

  {
    std::basic_ostringstream<char> os;
    std::basic_osyncstream<char> stream{os};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.get_wrapped();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.rdbuf();
  }
}
