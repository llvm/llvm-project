//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that format functions are marked [[nodiscard]] as a conforming extension

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <format>

#include "test_macros.h"

#ifndef TEST_HAS_NO_LOCALIZATION
#  include <locale>
#endif

void test() {
  // clang-format off
  std::format(""); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::vformat("", std::make_format_args()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::formatted_size(""); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_format_args(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::format(L""); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::vformat(L"", std::make_wformat_args()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::formatted_size(L""); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_wformat_args(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif // TEST_HAS_NO_WIDE_CHARACTERS

#ifndef TEST_HAS_NO_LOCALIZATION
  std::format(std::locale::classic(), ""); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::vformat(std::locale::classic(), "", std::make_format_args()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::formatted_size(std::locale::classic(), ""); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::format(std::locale::classic(), L""); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::vformat(std::locale::classic(), L"", std::make_wformat_args()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::formatted_size(std::locale::classic(), L""); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#  endif // TEST_HAS_NO_WIDE_CHARACTERS
#endif   // TEST_HAS_NO_LOCALIZATION
  // clang-format on
}
