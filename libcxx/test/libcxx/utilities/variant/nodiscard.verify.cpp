//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// Check that functions are marked [[nodiscard]]

#include <variant>

#include <test_macros.h>

void test() {
  {
    std::bad_variant_access ex;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ex.what();
  }

  {
    struct First {};
    struct Second {};

    std::variant<First, Second> v;
    const std::variant<First, Second> cv;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    cv.valueless_by_exception();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    cv.index();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::holds_alternative<First>(v);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get<0>(v);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get<0>(cv);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get<0>(std::move(v));
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get<0>(std::move(cv));
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get<First>(v);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get<First>(cv);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get<First>(std::move(v));
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get<First>(std::move(cv));

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get_if<0>(&v);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get_if<0>(&cv);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get_if<First>(&v);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get_if<First>(&cv);
  }

  {
    std::hash<std::variant<int, float>> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(std::variant<int, float>{});
  }
}
