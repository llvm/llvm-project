//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that functions are marked [[nodiscard]]

#include <stdexcept>
#include <system_error>

void test() {
  { // <stdexcept>
    std::logic_error le("logic error");
    le.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::runtime_error re("runtime error");
    re.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::domain_error de("domain error");
    de.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::invalid_argument ia("invalid argument");
    ia.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::length_error lerr("length error");
    lerr.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::out_of_range oor("out of range");
    oor.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::range_error rerr("range error");
    rerr.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::overflow_error oferr("overflow error");
    oferr.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::underflow_error uferr("underflow error");
    uferr.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  { // <system_error>
    {
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      std::generic_category();

      const std::error_category& ec = std::generic_category();

      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.name();
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.default_error_condition(94);
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.equivalent(94, ec.default_error_condition(82));
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.equivalent(std::error_code(49, ec), 94);
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.message(82);
    }
    {
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      std::system_category();

      const std::error_category& ec = std::system_category();

      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.name();
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.default_error_condition(94);
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.equivalent(94, ec.default_error_condition(82));
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.equivalent(std::error_code(49, ec), 94);
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.message(82);
    }
    {
      std::error_code ec;

      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.value();
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.category();
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.default_error_condition();

      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.message();

      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      std::make_error_code(std::errc::invalid_argument);
    }
    {
      std::error_condition ec;

      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.value();
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.category();
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      ec.message();

      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      std::make_error_condition(std::errc::invalid_argument);
    }
  }
}
