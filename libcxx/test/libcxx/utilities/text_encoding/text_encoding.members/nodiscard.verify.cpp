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

#include <text_encoding>

int main(int, char**) {
  std::text_encoding te = std::text_encoding();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.mib();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.name();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.aliases();

  // Clang does not emit a nodiscard warning for consteval functions with [[nodiscard]]: See issue #141536
  // expected-warning@+1 {{expression result unused}}
  std::text_encoding::literal();

  return 0;
}
