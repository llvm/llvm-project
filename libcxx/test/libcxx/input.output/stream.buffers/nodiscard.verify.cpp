//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <streambuf>

// Check that functions are marked [[nodiscard]]

#include <streambuf>

void test() {
  struct testbuf : public std::basic_streambuf<char> {
  } sbuf;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sbuf.getloc();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sbuf.in_avail();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sbuf.snextc();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sbuf.sgetc();
}
