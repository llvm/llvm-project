//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// Check that functions are marked [[nodiscard]]

#include <fstream>

#include "test_macros.h"

void test() {
  {
    std::basic_filebuf<char> fb;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    fb.is_open();
#if TEST_STD_VER >= 26
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    fb.native_handle();
#endif
  }

  {
    std::basic_ifstream<char> stream;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.rdbuf();
#if TEST_STD_VER >= 26
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.native_handle();
#endif
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.is_open();
  }

  {
    std::basic_ofstream<char> stream;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.rdbuf();
#if TEST_STD_VER >= 26
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.native_handle();
#endif
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.is_open();
  }

  {
    std::basic_fstream<char> stream;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.rdbuf();
#if TEST_STD_VER >= 26
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.native_handle();
#endif
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.is_open();
  }
}
