//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: availability-fp_to_chars-missing

// <format>

// template<ranges::input_range R, class charT>
//   requires (K == range_format::string || K == range_format::debug_string)
// struct range-default-formatter<K, R, charT>
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
                   "\nFormat string   ", fmt.get(), "\nExpected output ", expected, "\nActual output   ", out, '\n'));
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
