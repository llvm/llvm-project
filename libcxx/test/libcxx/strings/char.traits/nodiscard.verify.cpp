//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that functions are marked [[nodiscard]]

#include <string>

#include "test_macros.h"

void test_char() {
  typedef char char_t;
  typedef std::char_traits<char_t> traits;
  typedef typename traits::int_type int_t;

  const char_t buf[1] = {};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::eq(char_t(), char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::lt(char_t(), char_t());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::compare(buf, buf, 0);
  traits::length(buf); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::find(buf, 0, char_t());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::not_eof(char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::to_char_type(int_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::to_int_type(char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::eq_int_type(int_t(), int_t());
  traits::eof(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

#ifndef TEST_HAS_NO_CHAR8_T
void test_char8_t() {
  typedef char8_t char_t;
  typedef std::char_traits<char_t> traits;
  typedef typename traits::int_type int_t;

  const char_t buf[1] = {};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::eq(char_t(), char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::lt(char_t(), char_t());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::compare(buf, buf, 0);
  traits::length(buf); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::find(buf, 0, char_t());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::not_eof(char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::to_char_type(int_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::to_int_type(char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::eq_int_type(int_t(), int_t());
  traits::eof(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
#endif

void test_char16_t() {
  typedef char16_t char_t;
  typedef std::char_traits<char_t> traits;
  typedef typename traits::int_type int_t;

  const char_t buf[1] = {};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::eq(char_t(), char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::lt(char_t(), char_t());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::compare(buf, buf, 0);
  traits::length(buf); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::find(buf, 0, char_t());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::not_eof(char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::to_char_type(int_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::to_int_type(char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::eq_int_type(int_t(), int_t());
  traits::eof(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void test_char32_t() {
  typedef char32_t char_t;
  typedef std::char_traits<char_t> traits;
  typedef typename traits::int_type int_t;

  const char_t buf[1] = {};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::eq(char_t(), char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::lt(char_t(), char_t());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::compare(buf, buf, 0);
  traits::length(buf); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::find(buf, 0, char_t());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::not_eof(char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::to_char_type(int_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::to_int_type(char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::eq_int_type(int_t(), int_t());
  traits::eof(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
void test_wchar_t() {
  typedef wchar_t char_t;
  typedef std::char_traits<char_t> traits;
  typedef typename traits::int_type int_t;

  const char_t buf[1] = {};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::eq(char_t(), char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::lt(char_t(), char_t());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::compare(buf, buf, 0);
  traits::length(buf); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::find(buf, 0, char_t());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::not_eof(char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::to_char_type(int_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::to_int_type(char_t());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  traits::eq_int_type(int_t(), int_t());
  traits::eof(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
#endif
