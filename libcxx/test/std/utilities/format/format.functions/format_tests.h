//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_UTILITIES_FORMAT_FORMAT_FUNCTIONS_FORMAT_TESTS_H
#define TEST_STD_UTILITIES_FORMAT_FORMAT_FUNCTIONS_FORMAT_TESTS_H

#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)

// TestFunction must be callable as check(expected-result, string-to-format, args-to-format...)
// ExceptionTest must be callable as check_exception(expected-exception, string-to-format, args-to-format...)
template <class CharT, class TestFunction, class ExceptionTest>
void format_tests(TestFunction check, ExceptionTest check_exception) {
  // *** Test escaping  ***
  check(STR("{"), STR("{{"));
  check(STR("}"), STR("}}"));

  // *** Test argument ID ***
  check(STR("hello 01"), STR("hello {0:}{1:}"), false, true);
  check(STR("hello 10"), STR("hello {1:}{0:}"), false, true);

  // ** Test invalid format strings ***
  check_exception("The format string terminates at a '{'", STR("{"));
  check_exception("The replacement field misses a terminating '}'", STR("{:"),
                  42);

  check_exception("The format string contains an invalid escape sequence",
                  STR("}"));
  check_exception("The format string contains an invalid escape sequence",
                  STR("{:}-}"), 42);

  check_exception("The format string contains an invalid escape sequence",
                  STR("} "));

  check_exception(
      "The arg-id of the format-spec starts with an invalid character",
      STR("{-"), 42);
  check_exception("Argument index out of bounds", STR("hello {}"));
  check_exception("Argument index out of bounds", STR("hello {0}"));
  check_exception("Argument index out of bounds", STR("hello {1}"), 42);

  // *** Test char format argument ***
  // The `char` to `wchar_t` formatting is tested separately.
  check(STR("hello 09azAZ!"), STR("hello {}{}{}{}{}{}{}"), CharT('0'),
        CharT('9'), CharT('a'), CharT('z'), CharT('A'), CharT('Z'), CharT('!'));

  // *** Test string format argument ***
  {
    CharT buffer[] = {CharT('0'), CharT('9'), CharT('a'), CharT('z'),
                      CharT('A'), CharT('Z'), CharT('!'), 0};
    CharT* data = buffer;
    check(STR("hello 09azAZ!"), STR("hello {}"), data);
  }
  {
    CharT buffer[] = {CharT('0'), CharT('9'), CharT('a'), CharT('z'),
                      CharT('A'), CharT('Z'), CharT('!'), 0};
    const CharT* data = buffer;
    check(STR("hello 09azAZ!"), STR("hello {}"), data);
  }
  {
    std::basic_string<CharT> data = STR("world");
    check(STR("hello world"), STR("hello {}"), data);
  }
  {
    std::basic_string<CharT> buffer = STR("world");
    std::basic_string_view<CharT> data = buffer;
    check(STR("hello world"), STR("hello {}"), data);
  }

  // *** Test Boolean format argument ***
  check(STR("hello 01"), STR("hello {}{}"), false, true);

  // *** Test signed integral format argument ***
  check(STR("hello 42"), STR("hello {}"), static_cast<signed char>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<short>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<int>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<long>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<long long>(42));
#ifndef _LIBCPP_HAS_NO_INT128
  check(STR("hello 42"), STR("hello {}"), static_cast<__int128_t>(42));
  {
    // Note 128-bit support is only partly implemented test the range
    // conditions here.
    std::basic_string<CharT> min =
        std::format(STR("{}"), std::numeric_limits<long long>::min());
    check(min, STR("{}"),
          static_cast<__int128_t>(std::numeric_limits<long long>::min()));
    std::basic_string<CharT> max =
        std::format(STR("{}"), std::numeric_limits<long long>::max());
    check(max, STR("{}"),
          static_cast<__int128_t>(std::numeric_limits<long long>::max()));
    check_exception(
        "128-bit value is outside of implemented range", STR("{}"),
        static_cast<__int128_t>(std::numeric_limits<long long>::min()) - 1);
    check_exception(
        "128-bit value is outside of implemented range", STR("{}"),
        static_cast<__int128_t>(std::numeric_limits<long long>::max()) + 1);
  }
#endif

  // ** Test unsigned integral format argument ***
  check(STR("hello 42"), STR("hello {}"), static_cast<unsigned char>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<unsigned short>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<unsigned>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<unsigned long>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<unsigned long long>(42));
#ifndef _LIBCPP_HAS_NO_INT128
  check(STR("hello 42"), STR("hello {}"), static_cast<__uint128_t>(42));
  {
    // Note 128-bit support is only partly implemented test the range
    // conditions here.
    std::basic_string<CharT> max =
        std::format(STR("{}"), std::numeric_limits<unsigned long long>::max());
    check(max, STR("{}"),
          static_cast<__uint128_t>(
              std::numeric_limits<unsigned long long>::max()));
    check_exception("128-bit value is outside of implemented range", STR("{}"),
                    static_cast<__uint128_t>(
                        std::numeric_limits<unsigned long long>::max()) +
                        1);
  }
#endif

  // *** Test floating point format argument ***
// TODO FMT Enable after floating-point support has been enabled
#if 0
  check(STR("hello 42.000000"), STR("hello {}"), static_cast<float>(42));
  check(STR("hello 42.000000"), STR("hello {}"), static_cast<double>(42));
  check(STR("hello 42.000000"), STR("hello {}"), static_cast<long double>(42));
#endif
}

template <class TestFunction>
void format_tests_char_to_wchar_t(TestFunction check) {
  using CharT = wchar_t;
  check(STR("hello 09azA"), STR("hello {}{}{}{}{}"), '0', '9', 'a', 'z', 'A');
}

#endif
