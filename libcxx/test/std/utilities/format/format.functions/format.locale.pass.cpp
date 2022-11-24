//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format
// TODO FMT Evaluate gcc-12 status
// UNSUPPORTED:gcc-12

// <format>

// template<class... Args>
//   string format(const locale& loc, format-string<Args...> fmt, const Args&... args);
// template<class... Args>
//   wstring format(const locale& loc, wformat-string<Args...> fmt, const Args&... args);

#include <format>
#include <cassert>
#include <iostream>
#include <vector>

#include "test_macros.h"
#include "format_tests.h"
#include "string_literal.h"
#include "test_format_string.h"

auto test =
    []<class CharT, class... Args>(
        std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) constexpr {
      std::basic_string<CharT> out = std::format(std::locale(), fmt, std::forward<Args>(args)...);
      if constexpr (std::same_as<CharT, char>)
        if (out != expected)
          std::cerr << "\nFormat string   " << fmt.get() << "\nExpected output " << expected << "\nActual output   "
                    << out << '\n';
      assert(out == expected);
    };

auto test_exception = []<class CharT, class... Args>(std::string_view, std::basic_string_view<CharT>, Args&&...) {
  // After P2216 most exceptions thrown by std::format become ill-formed.
  // Therefore this tests does nothing.
  // A basic ill-formed test is done in format.locale.verify.cpp
  // The exceptions are tested by other functions that don't use the basic-format-string as fmt argument.
};

int main(int, char**) {
  format_tests<char>(test, test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests_char_to_wchar_t(test);
  format_tests<wchar_t>(test, test_exception);
#endif

  return 0;
}
