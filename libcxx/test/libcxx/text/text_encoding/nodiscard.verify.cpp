//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// UNSUPPORTED: no-localization

// <text_encoding>

#include <functional>
#include <text_encoding>

#include "test_macros.h"

int main(int, char**) {
  std::text_encoding te = std::text_encoding();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.mib();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.name();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.aliases();

#ifndef TEST_HAS_NO_LOCALIZATION
  te.environment();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.environment_is<std::text_encoding::UTF8>();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
  te.aliases();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}

  auto alias = te.aliases();

  alias.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  alias.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  auto it = alias.begin();

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

  std::hash<std::text_encoding>()(std::text_encoding::id::ASCII);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Clang does not emit a nodiscard warning for consteval functions with [[nodiscard]]: See issue #141536
  // expected-warning@+1 {{expression result unused}}
  std::text_encoding::literal();

  return 0;
}
