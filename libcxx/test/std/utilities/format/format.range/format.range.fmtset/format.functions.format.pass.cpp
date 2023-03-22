//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT Fix this test using GCC, it currently times out.
// UNSUPPORTED: gcc-12

// This test requires std::to_chars(floating-point), which is in the dylib
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0}}

// <format>

// template<ranges::input_range R, class charT>
//   struct range-default-formatter<range_format::set, R, charT>
//
// template<class... Args>
//   string format(format_string<Args...> fmt, Args&&... args);
// template<class... Args>
//   wstring format(wformat_string<Args...> fmt, Args&&... args);

#include <format>
#include <cassert>

#include "format.functions.tests.h"
#include "test_format_string.h"
#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

auto test = []<class CharT, class... Args>(
                std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) {
  std::basic_string<CharT> out = std::format(fmt, std::forward<Args>(args)...);
  TEST_REQUIRE(out == expected,
               TEST_WRITE_CONCATENATED(
                   "\nFormat string   ", fmt, "\nExpected output ", expected, "\nActual output   ", out, '\n'));
};

auto test_exception = []<class CharT, class... Args>(std::string_view, std::basic_string_view<CharT>, Args&&...) {
  // After P2216 most exceptions thrown by std::format become ill-formed.
  // Therefore this tests does nothing.
};

int main(int, char**) {
  format_tests<char>(test, test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests<wchar_t>(test, test_exception);
#endif

  return 0;
}
