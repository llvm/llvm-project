//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <expected>

// Test that ignoring std::expected generates [[nodiscard]] warnings.

#include <expected>

std::expected<int, int> returns_expected() { return std::expected<int, int>(5); }

std::expected<void, int> returns_expected_void() { return std::expected<void, int>(); }

void test() {
  returns_expected(); // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}

  returns_expected_void(); // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
}
