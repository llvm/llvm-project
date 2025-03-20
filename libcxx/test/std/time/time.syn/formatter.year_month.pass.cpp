//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// template<class charT> struct formatter<chrono::year_month, charT>;

#include <chrono>
#include <format>

#include <cassert>
#include <concepts>
#include <locale>
#include <iostream>
#include <type_traits>

#include "formatter_tests.h"
#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "string_literal.h"
#include "test_macros.h"

template <class CharT>
static void test_no_chrono_specs() {
  // Valid month
  check(SV("1970/Jan"), SV("{}"), std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{1}});
  check(SV("*1970/Jan*"), SV("{:*^10}"), std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{1}});
  check(SV("*1970/Jan"), SV("{:*>9}"), std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{1}});

  // Invalid month_day
  check(SV("1970/0 is not a valid month"),
        SV("{}"),
        std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{0}});
  check(SV("*1970/0 is not a valid month*"),
        SV("{:*^29}"),
        std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{0}});
}

template <class CharT>
static void test_invalid_values() {
  // Test that %b and %B throw an exception.
  check_exception("Formatting a month name from an invalid month number",
                  SV("{:%b}"),
                  std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{0}});

  check_exception("Formatting a month name from an invalid month number",
                  SV("{:%B}"),
                  std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{0}});
}

template <class CharT>
static void test_valid_values() {
  constexpr std::basic_string_view<CharT> fmt = SV(
      "{:"
      "%%b='%b'%t"
      "%%B='%B'%t"
      "%%C='%C'%t"
      "%%h='%h'%t"
      "%%y='%y'%t"
      "%%Y='%Y'%t"
      "%%EC='%EC'%t"
      "%%Ey='%Ey'%t"
      "%%EY='%EY'%t"
      "%%Oy='%Oy'%t"
      "%n}");

  constexpr std::basic_string_view<CharT> lfmt = SV(
      "{:L"
      "%%b='%b'%t"
      "%%B='%B'%t"
      "%%C='%C'%t"
      "%%h='%h'%t"
      "%%y='%y'%t"
      "%%Y='%Y'%t"
      "%%EC='%EC'%t"
      "%%Ey='%Ey'%t"
      "%%EY='%EY'%t"
      "%%Oy='%Oy'%t"
      "%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%b='Jan'\t"
           "%B='January'\t"
           "%C='19'\t"
           "%h='Jan'\t"
           "%y='70'\t"
           "%Y='1970'\t"
           "%EC='19'\t"
           "%Ey='70'\t"
           "%EY='1970'\t"
           "%Oy='70'\t"
           "\n"),
        fmt,
        std::chrono::year_month{std::chrono::year{1970}, std::chrono::January});

  check(SV("%b='May'\t"
           "%B='May'\t"
           "%C='20'\t"
           "%h='May'\t"
           "%y='04'\t"
           "%Y='2004'\t"
           "%EC='20'\t"
           "%Ey='04'\t"
           "%EY='2004'\t"
           "%Oy='04'\t"
           "\n"),
        fmt,
        std::chrono::year_month{std::chrono::year{2004}, std::chrono::May});

  // Use the global locale (fr_FR)
  check(SV(
#if defined(__APPLE__)
            "%b='jan'\t"
#else
            "%b='janv.'\t"
#endif
            "%B='janvier'\t"
            "%C='19'\t"
#if defined(__APPLE__)
            "%h='jan'\t"
#else
            "%h='janv.'\t"
#endif
            "%y='70'\t"
            "%Y='1970'\t"
            "%EC='19'\t"
            "%Ey='70'\t"
            "%EY='1970'\t"
            "%Oy='70'\t"
            "\n"),
        lfmt,
        std::chrono::year_month{std::chrono::year{1970}, std::chrono::January});

  check(SV("%b='mai'\t"
           "%B='mai'\t"
           "%C='20'\t"
           "%h='mai'\t"
           "%y='04'\t"
           "%Y='2004'\t"
           "%EC='20'\t"
           "%Ey='04'\t"
           "%EY='2004'\t"
           "%Oy='04'\t"
           "\n"),
        lfmt,
        std::chrono::year_month{std::chrono::year{2004}, std::chrono::May});

  // Use supplied locale (ja_JP)
  check(loc,
        SV(
#if defined(_WIN32)
            "%b='1'\t"
#elif defined(_AIX)      // defined(_WIN32)
            "%b='1月'\t"
#elif defined(__APPLE__) // defined(_WIN32)
            "%b=' 1'\t"
#else                    // defined(_WIN32)
            "%b=' 1月'\t"
#endif                   // defined(_WIN32)
            "%B='1月'\t"
            "%C='19'\t"
#if defined(_WIN32)
            "%h='1'\t"
#elif defined(_AIX)      // defined(_WIN32)
            "%h='1月'\t"
#elif defined(__APPLE__) // defined(_WIN32)
            "%h=' 1'\t"
#else                    // defined(_WIN32)
            "%h=' 1月'\t"
#endif                   // defined(_WIN32)
            "%y='70'\t"
            "%Y='1970'\t"
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
            "%EC='19'\t"
            "%Ey='70'\t"
            "%EY='1970'\t"
            "%Oy='70'\t"
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
            "%EC='昭和'\t"
            "%Ey='45'\t"
            "%EY='昭和45年'\t"
            "%Oy='七十'\t"
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
            "\n"),
        lfmt,
        std::chrono::year_month{std::chrono::year{1970}, std::chrono::January});

  check(loc,
        SV(

#if defined(_WIN32)
            "%b='5'\t"
#elif defined(_AIX)      // defined(_WIN32)
            "%b='5月'\t"
#elif defined(__APPLE__) // defined(_WIN32)
            "%b=' 5'\t"
#else                    // defined(_WIN32)
            "%b=' 5月'\t"
#endif                   // defined(_WIN32)
            "%B='5月'\t"
            "%C='20'\t"
#if defined(_WIN32)
            "%h='5'\t"
#elif defined(_AIX)      // defined(_WIN32)
            "%h='5月'\t"
#elif defined(__APPLE__) // defined(_WIN32)
            "%h=' 5'\t"
#else                    // defined(_WIN32)
            "%h=' 5月'\t"
#endif                   // defined(_WIN32)
            "%y='04'\t"
            "%Y='2004'\t"
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
            "%EC='20'\t"
            "%Ey='04'\t"
            "%EY='2004'\t"
            "%Oy='04'\t"
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
            "%EC='平成'\t"
            "%Ey='16'\t"
            "%EY='平成16年'\t"
            "%Oy='四'\t"
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
            "\n"),
        lfmt,
        std::chrono::year_month{std::chrono::year{2004}, std::chrono::May});

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test() {
  test_no_chrono_specs<CharT>();
  test_invalid_values<CharT>();
  test_valid_values<CharT>();

  check_invalid_types<CharT>(
      {SV("b"), SV("B"), SV("C"), SV("EC"), SV("Ey"), SV("EY"), SV("h"), SV("m"), SV("Om"), SV("Oy"), SV("y"), SV("Y")},
      std::chrono::year_month{std::chrono::year{1970}, std::chrono::January});

  check_exception("The format specifier expects a '%' or a '}'",
                  SV("{:A"),
                  std::chrono::year_month{std::chrono::year{1970}, std::chrono::January});
  check_exception("The chrono specifiers contain a '{'",
                  SV("{:%%{"),
                  std::chrono::year_month{std::chrono::year{1970}, std::chrono::January});
  check_exception("End of input while parsing a conversion specifier",
                  SV("{:%"),
                  std::chrono::year_month{std::chrono::year{1970}, std::chrono::January});
  check_exception("End of input while parsing the modifier E",
                  SV("{:%E"),
                  std::chrono::year_month{std::chrono::year{1970}, std::chrono::January});
  check_exception("End of input while parsing the modifier O",
                  SV("{:%O"),
                  std::chrono::year_month{std::chrono::year{1970}, std::chrono::January});

  // Precision not allowed
  check_exception("The format specifier expects a '%' or a '}'",
                  SV("{:.3}"),
                  std::chrono::year_month{std::chrono::year{1970}, std::chrono::January});
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
