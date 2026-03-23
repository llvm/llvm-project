//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <istream>
// <ostream>
// <iostream>

// Check that functions are marked [[nodiscard]]

#include <iostream>

void test() {
  struct testbuf : public std::basic_streambuf<char> {
  } sbuf;

  // [iostreamclass]

  // [istream]

  {
    std::basic_istream<char> stream(&sbuf);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.gcount();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.tellg();
  }

  // [ostream]

  {
    std::basic_ostream<char> stream(&sbuf);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.tellp();
  }
}
