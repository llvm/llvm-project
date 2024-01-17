//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: availability-fp_to_chars-missing

// <format>

// Tests the behavior of
//
// runtime-format-string<char> runtime_format(string_view fmt) noexcept;
// runtime-format-string<wchar_t> runtime_format(wstring_view fmt) noexcept;
//
// and
//
// template<class charT, class... Args>
//   struct basic_format_string {
//   ...
//   basic_format_string(runtime-format-string<charT> s) noexcept : str(s.str) {}
//   ...
// }
//
// This is done by testing it in the top-level functions:
//
// template<class... Args>
//   string format(format_string<Args...> fmt, Args&&... args);
// template<class... Args>
//   wstring format(wformat_string<Args...> fmt, Args&&... args);
//
// The basics of runtime_format and basic_format_string's constructor are tested in
// - libcxx/test/std/utilities/format/format.syn/runtime_format_string.pass.cpp
// - libcxx/test/std/utilities/format/format.fmt.string/ctor.runtime-format-string.pass.cpp

#include <format>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "format_tests.h"
#include "string_literal.h"
#include "assert_macros.h"
#include "concat_macros.h"

auto test = []<class CharT, class... Args>(
                std::basic_string_view<CharT> expected, std::basic_string_view<CharT> fmt, Args&&... args) constexpr {
  std::basic_string<CharT> out = std::format(std::runtime_format(fmt), std::forward<Args>(args)...);
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
          TEST_IGNORE_NODISCARD std::format(std::runtime_format(fmt), std::forward<Args>(args)...));
    };

int main(int, char**) {
  format_tests<char, execution_modus::partial>(test, test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests_char_to_wchar_t(test);
  format_tests<wchar_t, execution_modus::partial>(test, test_exception);
#endif

  return 0;
}
