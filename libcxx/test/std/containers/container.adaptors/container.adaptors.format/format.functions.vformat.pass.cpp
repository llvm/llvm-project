//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: availability-fp_to_chars-missing

// [container.adaptors.format]
// For each of queue, priority_queue, and stack, the library provides the
// following formatter specialization where adaptor-type is the name of the
// template:
//
// template<class charT, class T, formattable<charT> Container, class... U>
//   struct formatter<adaptor-type<T, Container, U...>, charT>

// string vformat(string_view fmt, format_args args);
// wstring vformat(wstring_view fmt, wformat_args args);

#include <format>
#include <cassert>

#include "format.functions.tests.h"
#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

auto test = []<class CharT, class... Args>(
                std::basic_string_view<CharT> expected, std::basic_string_view<CharT> fmt, Args&&... args) {
  std::basic_string<CharT> out = std::vformat(fmt, std::make_format_args<context_t<CharT>>(args...));
  TEST_REQUIRE(out == expected,
               TEST_WRITE_CONCATENATED(
                   "\nFormat string   ", fmt, "\nExpected output ", expected, "\nActual output   ", out, '\n'));
};

auto test_exception =
    []<class CharT, class... Args>(
        [[maybe_unused]] std::string_view what,
        [[maybe_unused]] std::basic_string_view<CharT> fmt,
        [[maybe_unused]] Args&&... args) {
      TEST_VALIDATE_EXCEPTION(
          std::format_error,
          [&]([[maybe_unused]] const std::format_error& e) {
            TEST_LIBCPP_REQUIRE(
                e.what() == what,
                TEST_WRITE_CONCATENATED(
                    "\nFormat string   ", fmt, "\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
          },
          TEST_IGNORE_NODISCARD std::vformat(fmt, std::make_format_args<context_t<CharT>>(args...)));
    };

int main(int, char**) {
  format_tests<char>(test, test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests<wchar_t>(test, test_exception);
#endif

  return 0;
}
