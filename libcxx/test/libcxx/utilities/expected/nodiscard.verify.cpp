//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <expected>

// Check that functions are marked [[nodiscard]]

#include <expected>
#include <utility>

void test() {
  // [expected.bad.void]

  class VoidBadExpectedAccess : public std::bad_expected_access<void> {};

  VoidBadExpectedAccess voidEx;
  const VoidBadExpectedAccess cVoidEx{};

  voidEx.what();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cVoidEx.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // [expected.bad]

  std::bad_expected_access<char> ex{'z'};
  const std::bad_expected_access<char> cEx{'z'};

  ex.error();             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cEx.error();            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(ex).error();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cEx).error(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // [expected.expected]

  std::expected<int, int> exp;
  const std::expected<int, int> cExp{};

  *cExp;            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  *exp;             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::move(cExp); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::move(exp);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  exp.has_value(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  cExp.value();            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  exp.value();             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cExp).value(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(exp).value();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  cExp.error();            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  exp.error();             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cExp).error(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(exp).error();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cExp.value_or(94);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(exp).value_or(94);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cExp.error_or(82);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(exp).error_or(82);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  exp.and_then([](int&) { return std::expected<int, int>{94}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cExp.and_then([](const int&) { return std::expected<int, int>{94}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(exp).and_then([](int&&) { return std::expected<int, int>{94}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cExp).and_then([](const int&&) { return std::expected<int, int>{94}; });

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  exp.or_else([](int&) { return std::expected<int, int>{82}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cExp.or_else([](const int&) { return std::expected<int, int>{82}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(exp).or_else([](int&&) { return std::expected<int, int>{82}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cExp).or_else([](const int&&) { return std::expected<int, int>{82}; });

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  exp.transform([](int) { return 94; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cExp.transform([](const int) { return 94; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(exp).transform([](int&&) { return 94; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cExp).transform([](const int&&) { return 94; });

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  exp.transform_error([](int) { return 82; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cExp.transform_error([](const int) { return 82; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(exp).transform_error([](int&&) { return 82; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cExp).transform_error([](const int&&) { return 82; });

  // [expected.void]

  std::expected<void, int> vExp;
  const std::expected<void, int> cVExp{};

  vExp.has_value(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cVExp.error();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  vExp.error();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cVExp).error();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(vExp).error();

  vExp.error_or(94);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cVExp.error_or(94); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  vExp.and_then([]() -> std::expected<int, int> { return 82; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cVExp.and_then([]() -> std::expected<int, int> { return 82; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(vExp).and_then([]() -> std::expected<int, int> { return 82; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cVExp).and_then([]() -> std::expected<int, int> { return 82; });

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  vExp.or_else([](auto) -> std::expected<void, long> { return {}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cVExp.or_else([](auto) -> std::expected<void, long> { return {}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(vExp).or_else([](auto) -> std::expected<void, long> { return {}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cVExp).or_else([](auto) -> std::expected<void, long> { return {}; });

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  vExp.transform([]() -> int { return 94; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cVExp.transform([]() -> int { return 94; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(vExp).transform([]() -> int { return 94; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cVExp).transform([]() -> int { return 94; });

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  vExp.transform_error([](auto) -> int { return 82; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cVExp.transform_error([](auto) -> int { return 82; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(vExp).transform_error([](auto) -> int { return 82; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cVExp).transform_error([](auto) -> int { return 82; });

  // [expected.unexpected]

  std::unexpected<char> unex('z');
  const std::unexpected<char> cUnex('z');

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  unex.error();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cUnex.error();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(unex).error();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cUnex).error();
}
