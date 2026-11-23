//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX26_REMOVED_STRSTREAM

// <strstream>

// Check that functions are marked [[nodiscard]]

#include <strstream>

void test() {
  {
    std::strstreambuf sb;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sb.str();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sb.pcount();
  }

  {
    const char buff[] = "";
    std::istrstream stream(buff);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.rdbuf();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.str();
  }

  {
    std::ostrstream stream;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.rdbuf();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.str();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.pcount();
  }

  {
    std::strstream stream;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.rdbuf();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.str();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    stream.pcount();
  }
}
