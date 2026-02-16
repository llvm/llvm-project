//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// Check that functions are marked [[nodiscard]]

#include <memory>
#include <sstream>
#include <utility>

#include "test_macros.h"

void test() {
#if TEST_STD_VER >= 20
  std::allocator<char> alloc;
#endif

  {
    std::basic_stringbuf<char> sb;

#if TEST_STD_VER >= 20
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sb.get_allocator();
#endif

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sb.str();
#if TEST_STD_VER >= 20
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(sb).str();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sb.str(alloc);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sb.view();
#endif
  }

  {
    std::basic_istringstream<char> stream;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.rdbuf();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.str();
#if TEST_STD_VER >= 20
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(stream).str();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.str(alloc);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.view();
#endif
  }

  {
    std::basic_ostringstream<char> stream;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.rdbuf();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.str();
#if TEST_STD_VER >= 20
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(stream).str();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.str(alloc);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.view();
#endif
  }

  {
    std::basic_stringstream<char> stream;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.rdbuf();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.str();
#if TEST_STD_VER >= 20
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(stream).str();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.str(alloc);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.view();
#endif
  }
}
