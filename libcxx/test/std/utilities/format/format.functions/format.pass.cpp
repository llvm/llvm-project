//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-format
// TODO FMT Evaluate gcc-11 status
// UNSUPPORTED: gcc-11

// Note this formatter shows additional information when tests are failing.
// This aids the development. Since other formatters fail in the same fashion
// they don't have this additional output.

// <format>

// template<class... Args>
//   string format(string_view fmt, const Args&... args);
// template<class... Args>
//   wstring format(wstring_view fmt, const Args&... args);

#include <format>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "format_tests.h"

#ifndef TEST_HAS_NO_LOCALIZATION
#  include <iostream>
#endif

auto test = []<class CharT, class... Args>(std::basic_string_view<CharT> expected, std::basic_string_view<CharT> fmt,
                                           const Args&... args) {
  std::basic_string<CharT> out = std::format(fmt, args...);
#ifndef TEST_HAS_NO_LOCALIZATION
  if constexpr (std::same_as<CharT, char>)
    if (out != expected)
      std::cerr << "\nFormat string   " << fmt << "\nExpected output " << expected << "\nActual output   " << out
                << '\n';
#endif
  assert(out == expected);
};

auto test_exception = []<class CharT, class... Args>(std::string_view what, std::basic_string_view<CharT> fmt,
                                                     const Args&... args) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    std::format(fmt, args...);
#  ifndef TEST_HAS_NO_LOCALIZATION
    if constexpr (std::same_as<CharT, char>)
      std::cerr << "\nFormat string   " << fmt << "\nDidn't throw an exception.\n";
#  endif
    assert(false);
  } catch (const std::format_error& e) {
    if constexpr (std::same_as<CharT, char>)
#  if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_LOCALIZATION)
      if (e.what() != what)
        std::cerr << "\nFormat string   " << fmt << "\nExpected exception " << what << "\nActual exception   "
                  << e.what() << '\n';
#  endif
    LIBCPP_ASSERT(e.what() == what);
    return;
  }
  assert(false);
#endif
  (void)what;
  (void)fmt;
  (void)sizeof...(args);
};

int main(int, char**) {
  format_tests<char>(test, test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests_char_to_wchar_t(test);
  format_tests<wchar_t>(test, test_exception);
#endif

  return 0;
}
