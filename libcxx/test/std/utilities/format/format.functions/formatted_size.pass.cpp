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

// <format>

// template<class... Args>
//   size_t formatted_size(string_view fmt, const Args&... args);
// template<class... Args>
//   size_t formatted_size(wstring_view fmt, const Args&... args);

#include <format>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "format_tests.h"

auto test = []<class CharT, class... Args>(std::basic_string_view<CharT> expected, std::basic_string_view<CharT> fmt,
                                           const Args&... args) {
  size_t size = std::formatted_size(fmt, args...);
  assert(size == expected.size());
};

auto test_exception = []<class CharT, class... Args>(std::string_view what, std::basic_string_view<CharT> fmt,
                                                     const Args&... args) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    std::formatted_size(fmt, args...);
    assert(false);
  } catch (const std::format_error& e) {
    LIBCPP_ASSERT(e.what() == what);
    return;
  }
  assert(false);
#else
  (void)what;
  (void)fmt;
  (void)sizeof...(args);
#endif
};

int main(int, char**) {
  format_tests<char>(test, test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests_char_to_wchar_t(test);
  format_tests<wchar_t>(test, test_exception);
#endif

  return 0;
}
