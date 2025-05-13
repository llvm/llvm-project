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

std::expected<int, int> returns_expected();
std::expected<void, int> returns_expected_void();

void test() {
  returns_expected(); // expected-warning {{ignoring return value of type 'expected<int, int>'}}
  returns_expected_void(); // expected-warning {{ignoring return value of type 'expected<void, int>'}}
}
