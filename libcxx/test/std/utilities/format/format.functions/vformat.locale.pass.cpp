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
// UNSUPPORTED: gcc-12
// TODO FMT Investigate AppleClang ICE
// UNSUPPORTED: apple-clang-13

// <format>

// string vformat(const locale& loc, string_view fmt, format_args args);
// wstring vformat(const locale& loc, wstring_view fmt, wformat_args args);

#include <format>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "format_tests.h"
#include "string_literal.h"

auto test = []<string_literal fmt, class CharT, class... Args>(std::basic_string_view<CharT> expected,
                                                               const Args&... args) constexpr {
  std::basic_string<CharT> out =
      std::vformat(std::locale(), fmt.template sv<CharT>(), std::make_format_args<context_t<CharT>>(args...));
  assert(out == expected);
};

auto test_exception = []<class CharT, class... Args>(std::string_view what, std::basic_string_view<CharT> fmt,
                                                     const Args&... args) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    (void)std::vformat(std::locale(), fmt, std::make_format_args<context_t<CharT>>(args...));
    assert(false);
  } catch ([[maybe_unused]] const std::format_error& e) {
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
