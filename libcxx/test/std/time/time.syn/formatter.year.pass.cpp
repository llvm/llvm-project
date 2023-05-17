//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-FREEBSD-FIXME

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT Investigate Windows issues.
// UNSUPPORTED: msvc, target={{.+}}-windows-gnu

// TODO FMT Fix this test using GCC, it currently crashes.
// UNSUPPORTED: gcc-12

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// template<class charT> struct formatter<chrono::year, charT>;

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
  check(SV("-32767"), SV("{}"), std::chrono::year{-32'767});
  check(SV("-1000"), SV("{}"), std::chrono::year{-1000});
  check(SV("-0100"), SV("{}"), std::chrono::year{-100});
  check(SV("-0010"), SV("{}"), std::chrono::year{-10});
  check(SV("-0001"), SV("{}"), std::chrono::year{-1});
  check(SV("0000"), SV("{}"), std::chrono::year{0});
  check(SV("0001"), SV("{}"), std::chrono::year{1});
  check(SV("0010"), SV("{}"), std::chrono::year{10});
  check(SV("0100"), SV("{}"), std::chrono::year{100});
  check(SV("1000"), SV("{}"), std::chrono::year{1000});
  check(SV("32727"), SV("{}"), std::chrono::year{32'727});

  // Invalid year
  check(SV("-32768 is not a valid year"), SV("{}"), std::chrono::year{-32'768});
  check(SV("-32768 is not a valid year"), SV("{}"), std::chrono::year{32'768});
}

template <class CharT>
static void test_valid_values() {
  constexpr std::basic_string_view<CharT> fmt = SV(
      "{:"
      "%%C='%C'%t"
      "%%EC='%EC'%t"
      "%%y='%y'%t"
      "%%Ey='%Ey'%t"
      "%%Oy='%Oy'%t"
      "%%Y='%Y'%t"
      "%%EY='%EY'%t"
      "%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV(
      "{:L"
      "%%C='%C'%t"
      "%%EC='%EC'%t"
      "%%y='%y'%t"
      "%%Ey='%Ey'%t"
      "%%Oy='%Oy'%t"
      "%%Y='%Y'%t"
      "%%EY='%EY'%t"
      "%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%C='00'\t"
#if defined(__APPLE__)
           "%EC='00'\t"
#else
           "%EC='0'\t"
#endif
           "%y='00'\t"
           "%Ey='00'\t"
           "%Oy='00'\t"
           "%Y='0000'\t"
#if defined(__APPLE__)
           "%EY='0000'\t"
#elif defined(_AIX)
           "%EY=''\t"
#else
           "%EY='0'\t"
#endif
           "\n"),
        fmt,
        std::chrono::year{0});

  check(SV("%C='19'\t"
           "%EC='19'\t"
           "%y='70'\t"
           "%Ey='70'\t"
           "%Oy='70'\t"
           "%Y='1970'\t"
           "%EY='1970'\t"
           "\n"),
        fmt,
        std::chrono::year{1970});

  check(SV("%C='20'\t"
           "%EC='20'\t"
           "%y='38'\t"
           "%Ey='38'\t"
           "%Oy='38'\t"
           "%Y='2038'\t"
           "%EY='2038'\t"
           "\n"),
        fmt,
        std::chrono::year{2038});

  // Use the global locale (fr_FR)
  check(SV("%C='00'\t"
#if defined(__APPLE__)
           "%EC='00'\t"
#else
           "%EC='0'\t"
#endif
           "%y='00'\t"
           "%Ey='00'\t"
           "%Oy='00'\t"
           "%Y='0000'\t"
#if defined(__APPLE__)
           "%EY='0000'\t"
#elif defined(_AIX)
           "%EY=''\t"
#else
           "%EY='0'\t"
#endif
           "\n"),
        lfmt,
        std::chrono::year{0});

  check(SV("%C='19'\t"
           "%EC='19'\t"
           "%y='70'\t"
           "%Ey='70'\t"
           "%Oy='70'\t"
           "%Y='1970'\t"
           "%EY='1970'\t"
           "\n"),
        lfmt,
        std::chrono::year{1970});

  check(SV("%C='20'\t"
           "%EC='20'\t"
           "%y='38'\t"
           "%Ey='38'\t"
           "%Oy='38'\t"
           "%Y='2038'\t"
           "%EY='2038'\t"
           "\n"),
        lfmt,
        std::chrono::year{2038});

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(__APPLE__) || defined(_AIX)

  check(SV("%C='00'\t"
#  if defined(__APPLE__)
           "%EC='00'\t"
#  else
           "%EC='0'\t"
#  endif
           "%y='00'\t"
           "%Ey='00'\t"
           "%Oy='00'\t"
           "%Y='0000'\t"
#  if defined(_AIX)
           "%EY=''\t"
#  else
           "%EY='0000'\t"
#  endif
           "\n"),
        lfmt,
        std::chrono::year{0});

  check(SV("%C='19'\t"
           "%EC='19'\t"
           "%y='70'\t"
           "%Ey='70'\t"
           "%Oy='70'\t"
           "%Y='1970'\t"
           "%EY='1970'\t"
           "\n"),
        lfmt,
        std::chrono::year{1970});

  check(SV("%C='20'\t"
           "%EC='20'\t"
           "%y='38'\t"
           "%Ey='38'\t"
           "%Oy='38'\t"
           "%Y='2038'\t"
           "%EY='2038'\t"
           "\n"),
        lfmt,
        std::chrono::year{2038});

#else // defined(__APPLE__) || defined(_AIX)
  check(loc,
        SV("%C='00'\t"
           "%EC='紀元前'\t"
           "%y='00'\t"
// https://sourceware.org/bugzilla/show_bug.cgi?id=23758
#  if defined(__GLIBC__) && __GLIBC__ <= 2 && __GLIBC_MINOR__ < 29
           "%Ey='1'\t"
#  else
           "%Ey='01'\t"
#  endif
           "%Oy='〇'\t"
           "%Y='0000'\t"
// https://sourceware.org/bugzilla/show_bug.cgi?id=23758
#  if defined(__GLIBC__) && __GLIBC__ <= 2 && __GLIBC_MINOR__ < 29
           "%EY='紀元前1年'\t"
#  else
           "%EY='紀元前01年'\t"
#  endif
           "\n"),
        lfmt,
        std::chrono::year{0});

  check(loc,
        SV("%C='19'\t"
           "%EC='昭和'\t"
           "%y='70'\t"
           "%Ey='45'\t"
           "%Oy='七十'\t"
           "%Y='1970'\t"
           "%EY='昭和45年'\t"
           "\n"),
        lfmt,
        std::chrono::year{1970});

  // Note this test will fail if the Reiwa era ends before 2038.
  check(loc,
        SV("%C='20'\t"
           "%EC='令和'\t"
           "%y='38'\t"
           "%Ey='20'\t"
           "%Oy='三十八'\t"
           "%Y='2038'\t"
           "%EY='令和20年'\t"
           "\n"),
        lfmt,
        std::chrono::year{2038});
#endif // defined(__APPLE__) || defined(_AIX)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_padding() {
  constexpr std::basic_string_view<CharT> fmt = SV("{:%%C='%C'%t%%y='%y'%t%%Y='%Y'%t%n}");

  check(SV("%C='-100'\t%y='99'\t%Y='-9999'\t\n"), fmt, std::chrono::year{-9'999});
  check(SV("%C='-10'\t%y='99'\t%Y='-0999'\t\n"), fmt, std::chrono::year{-999});
  check(SV("%C='-1'\t%y='99'\t%Y='-0099'\t\n"), fmt, std::chrono::year{-99});
  check(SV("%C='-1'\t%y='09'\t%Y='-0009'\t\n"), fmt, std::chrono::year{-9});
  check(SV("%C='00'\t%y='00'\t%Y='0000'\t\n"), fmt, std::chrono::year{0});
  check(SV("%C='00'\t%y='09'\t%Y='0009'\t\n"), fmt, std::chrono::year{9});
  check(SV("%C='00'\t%y='99'\t%Y='0099'\t\n"), fmt, std::chrono::year{99});
  check(SV("%C='09'\t%y='99'\t%Y='0999'\t\n"), fmt, std::chrono::year{999});
  check(SV("%C='99'\t%y='99'\t%Y='9999'\t\n"), fmt, std::chrono::year{9'999});
  check(SV("%C='100'\t%y='00'\t%Y='10000'\t\n"), fmt, std::chrono::year{10'000});
}

template <class CharT>
static void test() {
  test_no_chrono_specs<CharT>();
  test_valid_values<CharT>();
  test_padding<CharT>();
  check_invalid_types<CharT>(
      {SV("C"), SV("y"), SV("Y"), SV("EC"), SV("Ey"), SV("EY"), SV("Oy")}, std::chrono::year{1970});

  check_exception("Expected '%' or '}' in the chrono format-string", SV("{:A"), std::chrono::year{1970});
  check_exception("The chrono-specs contains a '{'", SV("{:%%{"), std::chrono::year{1970});
  check_exception("End of input while parsing the modifier chrono conversion-spec", SV("{:%"), std::chrono::year{1970});
  check_exception("End of input while parsing the modifier E", SV("{:%E"), std::chrono::year{1970});
  check_exception("End of input while parsing the modifier O", SV("{:%O"), std::chrono::year{1970});

  // Precision not allowed
  check_exception("Expected '%' or '}' in the chrono format-string", SV("{:.3}"), std::chrono::year{1970});
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
