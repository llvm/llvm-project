//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// This version runs the test when the platform has Unicode support.
// UNSUPPORTED: libcpp-has-no-unicode

// XFAIL: availability-fp_to_chars-missing

// <format>

// The paper
//   P2572R1 std::format fill character allowances
// adds support for Unicode Scalar Values as fill character.

#include <format>

#include "assert_macros.h"
#include "concat_macros.h"
#include "format.functions.common.h"
#include "make_string.h"
#include "string_literal.h"
#include "test_format_string.h"
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

auto check = []<class CharT, class... Args>(
                 std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) {
  std::basic_string<CharT> out = std::format(fmt, std::forward<Args>(args)...);
  TEST_REQUIRE(out == expected,
               TEST_WRITE_CONCATENATED(
                   "\nFormat string   ", fmt.get(), "\nExpected output ", expected, "\nActual output   ", out, '\n'));
};

auto check_exception =
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

template <class CharT>
void test() {
  // 1, 2, 3, 4 code unit UTF-8 transitions
  check(SV("\u000042\u0000"), SV("{:\u0000^4}"), 42);
  check(SV("\u007f42\u007f"), SV("{:\u007f^4}"), 42);
  check(SV("\u008042\u0080"), SV("{:\u0080^4}"), 42);
  check(SV("\u07ff42\u07ff"), SV("{:\u07ff^4}"), 42);
  check(SV("\u080042\u0800"), SV("{:\u0800^4}"), 42);
  check(SV("\uffff42\uffff"), SV("{:\uffff^4}"), 42);
  check(SV("\U0010000042\U00100000"), SV("{:\U00100000^4}"), 42);
  check(SV("\U0010ffff42\U0010ffff"), SV("{:\U0010ffff^4}"), 42);

  // Examples of P2572R1
  check(SV("🤡🤡x🤡🤡🤡"), SV("{:🤡^6}"), SV("x"));
  check(SV("🤡🤡🤡"), SV("{:*^6}"), SV("🤡🤡🤡"));
  check(SV("12345678"), SV("{:*>6}"), SV("12345678"));

  // Invalid Unicode Scalar Values
  if constexpr (std::same_as<CharT, char>) {
    check_exception("The format specifier contains malformed Unicode characters", SV("{:\xed\xa0\x80^}"), 42); // U+D800
    check_exception("The format specifier contains malformed Unicode characters", SV("{:\xed\xa0\xbf^}"), 42); // U+DBFF
    check_exception("The format specifier contains malformed Unicode characters", SV("{:\xed\xbf\x80^}"), 42); // U+DC00
    check_exception("The format specifier contains malformed Unicode characters", SV("{:\xed\xbf\xbf^}"), 42); // U+DFFF

    check_exception(
        "The format specifier contains malformed Unicode characters", SV("{:\xf4\x90\x80\x80^}"), 42); // U+110000
    check_exception(
        "The format specifier contains malformed Unicode characters", SV("{:\xf4\x90\xbf\xbf^}"), 42); // U+11FFFF

    check_exception("The format specifier contains malformed Unicode characters",
                    SV("{:\x80^}"),
                    42); // Trailing code unit with no leading one.
    check_exception("The format specifier contains malformed Unicode characters",
                    SV("{:\xc0^}"),
                    42); // Missing trailing code unit.
    check_exception("The format specifier contains malformed Unicode characters",
                    SV("{:\xe0\x80^}"),
                    42); // Missing trailing code unit.
    check_exception("The format specifier contains malformed Unicode characters",
                    SV("{:\xf0\x80^}"),
                    42); // Missing two trailing code units.
    check_exception("The format specifier contains malformed Unicode characters",
                    SV("{:\xf0\x80\x80^}"),
                    42); // Missing trailing code unit.

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  } else {
#  ifdef TEST_SHORT_WCHAR
    check_exception("The format specifier contains malformed Unicode characters", std::wstring_view{L"{:\xd800^}"}, 42);
    check_exception("The format specifier contains malformed Unicode characters", std::wstring_view{L"{:\xdbff^}"}, 42);
    check_exception("The format specifier contains malformed Unicode characters", std::wstring_view{L"{:\xdc00^}"}, 42);
    check_exception("The format specifier contains malformed Unicode characters", std::wstring_view{L"{:\xddff^}"}, 42);

    check_exception("The format specifier contains malformed Unicode characters",
                    std::wstring_view{L"{:\xdc00\xd800^}"},
                    42); // Reverted surrogates.

#  else  // TEST_SHORT_WCHAR
    check_exception("The fill option contains an invalid value", std::wstring_view{L"{:\xd800^}"}, 42);
    check_exception("The fill option contains an invalid value", std::wstring_view{L"{:\xdbff^}"}, 42);
    check_exception("The fill option contains an invalid value", std::wstring_view{L"{:\xdc00^}"}, 42);
    check_exception("The fill option contains an invalid value", std::wstring_view{L"{:\xddff^}"}, 42);

    check_exception(
        "The format specifier should consume the input or end with a '}'", std::wstring_view{L"{:\xdc00\xd800^}"}, 42);

    check_exception("The fill option contains an invalid value", std::wstring_view{L"{:\x00110000^}"}, 42);
    check_exception("The fill option contains an invalid value", std::wstring_view{L"{:\x0011ffff^}"}, 42);
#  endif // TEST_SHORT_WCHAR
#endif   // TEST_HAS_NO_WIDE_CHARACTERS
  }
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
