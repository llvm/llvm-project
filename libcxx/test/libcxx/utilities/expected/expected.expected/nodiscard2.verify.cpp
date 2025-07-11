//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_NODISCARD_EXPECTED

// <expected>

// Test that ignoring std::expected does not generate [[nodiscard]] warnings.

// expected-no-diagnostics

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
  returns_expected();
  returns_expected_void();

  returns_expected().and_then(and_then);
  returns_expected_void().and_then(and_then_void);
  returns_expected().or_else(or_else);
  returns_expected_void().or_else(or_else_void);
  returns_expected().transform(transform);
  returns_expected_void().transform(transform_void);
  returns_expected().transform_error(transform_error);
  returns_expected_void().transform_error(transform_error_void);
}
