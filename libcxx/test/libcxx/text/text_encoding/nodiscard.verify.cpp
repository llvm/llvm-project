//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

#include <text_encoding>

#include "test_macros.h"

void test() {
  std::text_encoding te{};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.mib();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.name();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.aliases();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.aliases();

  auto alias = te.aliases();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  alias.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  alias.end();

  auto it = alias.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it[0];
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it + 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hash<std::text_encoding>()(std::text_encoding::ASCII);

#if !defined(TEST_HAS_NO_LOCALIZATION) && _LIBCPP_AVAILABILITY_HAS_TEXT_ENCODING_ENVIRONMENT
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.environment();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.environment_is<std::text_encoding::UTF8>();
#endif
}

consteval void literal() {
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::text_encoding::literal();
}
