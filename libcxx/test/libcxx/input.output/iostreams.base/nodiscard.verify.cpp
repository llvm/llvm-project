//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// Check that functions are marked [[nodiscard]]

#include <cwchar>
#include <ios>

#include "test_macros.h"

void test() {
  class test_stream : public std::ios {
  public:
    test_stream() { init(0); }
  };
  test_stream stream;

  {
    std::ios_base& ref = stream;

    ref.flags();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.precision(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.width();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.getloc();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ios_base::xalloc();
    ref.iword(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.pword(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // extensions
    ref.rdstate();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.good();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.eof();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.fail();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.bad();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.exceptions(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::ios& ref = stream;

    !ref;             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.rdstate();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.good();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.eof();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.fail();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.bad();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.exceptions(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    ref.tie();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.rdbuf(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    ref.fill(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ref.narrow('\0', '\0');
    ref.widen('\0'); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::fpos<std::mbstate_t> pos;

    pos.state(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    pos + std::streamoff(0);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    pos - std::streamoff(0);
    pos - pos; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::iostream_category();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_error_code(std::io_errc(1));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_error_condition(std::io_errc(1));
}
