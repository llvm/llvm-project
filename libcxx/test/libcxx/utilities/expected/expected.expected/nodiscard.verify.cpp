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

std::expected<int, int> and_then(int);
std::expected<void, int> and_then_void();
std::expected<int, int> or_else(int);
std::expected<void, int> or_else_void(int);
int transform(int);
void transform_void();
int transform_error(int);
int transform_error_void(int);

void test() {
  returns_expected();      // expected-warning {{ignoring return value of type 'expected<int, int>'}}
  returns_expected_void(); // expected-warning {{ignoring return value of type 'expected<void, int>'}}

  returns_expected().and_then(and_then);
      // expected-warning@-1 {{ignoring return value}}
  returns_expected_void().and_then(and_then_void);
      // expected-warning@-1 {{ignoring return value}}
  returns_expected().or_else(or_else);
      // expected-warning@-1 {{ignoring return value}}
  returns_expected_void().or_else(or_else_void);
      // expected-warning@-1 {{ignoring return value}}
  returns_expected().transform(transform);
      // expected-warning@-1 {{ignoring return value}}
  returns_expected_void().transform(transform_void);
      // expected-warning@-1 {{ignoring return value}}
  returns_expected().transform_error(transform_error);
      // expected-warning@-1 {{ignoring return value}}
  returns_expected_void().transform_error(transform_error_void);
      // expected-warning@-1 {{ignoring return value}}
}
