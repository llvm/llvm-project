//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-threads

// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// <thread>

// template<class charT>
// struct formatter<thread::id, charT>;

// string vformat(string_view fmt, format_args args);
// wstring vformat(wstring_view fmt, wformat_args args);

#include <cassert>
#include <format>
#include <thread>

#include "assert_macros.h"
#include "concat_macros.h"
#include "format.functions.common.h"
#include "test_macros.h"

template <class CharT, class ExceptionTest>
void format_tests(ExceptionTest check_exception) {
  // Note the output of std::thread::id is unspecified. The output text is the
  // same as the stream operator. Since that format is already released this
  // test follows the practice on existing systems.
  std::thread::id input{};

  /***** Test the type generic part *****/
  check_exception("The format string contains an invalid escape sequence", SV("{:}<}"), input);
  check_exception("The fill option contains an invalid value", SV("{:{<}"), input);

  // *** sign ***
  check_exception("The replacement field misses a terminating '}'", SV("{:-}"), input);
  check_exception("The replacement field misses a terminating '}'", SV("{:+}"), input);
  check_exception("The replacement field misses a terminating '}'", SV("{: }"), input);

  // *** alternate form ***
  check_exception("The replacement field misses a terminating '}'", SV("{:#}"), input);

  // *** zero-padding ***
  check_exception("The width option should not have a leading zero", SV("{:0}"), input);

  // *** precision ***
  check_exception("The replacement field misses a terminating '}'", SV("{:.}"), input);

  // *** locale-specific form ***
  check_exception("The replacement field misses a terminating '}'", SV("{:L}"), input);

  // *** type ***
  for (std::basic_string_view<CharT> fmt : fmt_invalid_types<CharT>(""))
    check_exception("The replacement field misses a terminating '}'", fmt, input);
}

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
  format_tests<char>(test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests<wchar_t>(test_exception);
#endif

  return 0;
}
