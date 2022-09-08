//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT Investigate Windows issues.
// UNSUPPORTED: msvc, target={{.+}}-windows-gnu

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// template<class charT> struct formatter<chrono::day, charT>;

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
  using namespace std::literals::chrono_literals;

  // Valid day
  check.template operator()<"{}">(SV("01"), 1d);
  check.template operator()<"{:*^4}">(SV("*01*"), 1d);
  check.template operator()<"{:*>3}">(SV("*01"), 1d);

  // Invalid day
  check.template operator()<"{}">(SV("00 is not a valid day"), 0d);
  check.template operator()<"{:*^23}">(SV("*00 is not a valid day*"), 0d);
}

template <class CharT>
static void test_valid_values() {
  using namespace std::literals::chrono_literals;

  constexpr string_literal fmt{"{:%%d='%d'%t%%Od='%Od'%t%%e='%e'%t%%Oe='%Oe'%n}"};
  constexpr string_literal lfmt{"{:L%%d='%d'%t%%Od='%Od'%t%%e='%e'%t%%Oe='%Oe'%n}"};

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check.template operator()<fmt>(SV("%d='00'\t%Od='00'\t%e=' 0'\t%Oe=' 0'\n"), 0d);
  check.template operator()<fmt>(SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"), 1d);
  check.template operator()<fmt>(SV("%d='31'\t%Od='31'\t%e='31'\t%Oe='31'\n"), 31d);
#if defined(_AIX)
  check.template operator()<fmt>(SV("%d='55'\t%Od='55'\t%e='55'\t%Oe='55'\n"), 255d);
#else
  check.template operator()<fmt>(SV("%d='255'\t%Od='255'\t%e='255'\t%Oe='255'\n"), 255d);
#endif

  // Use the global locale (fr_FR)
  check.template operator()<lfmt>(SV("%d='00'\t%Od='00'\t%e=' 0'\t%Oe=' 0'\n"), 0d);
  check.template operator()<lfmt>(SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"), 1d);
  check.template operator()<lfmt>(SV("%d='31'\t%Od='31'\t%e='31'\t%Oe='31'\n"), 31d);
#if defined(_AIX)
  check.template operator()<lfmt>(SV("%d='55'\t%Od='55'\t%e='55'\t%Oe='55'\n"), 255d);
#else
  check.template operator()<lfmt>(SV("%d='255'\t%Od='255'\t%e='255'\t%Oe='255'\n"), 255d);
#endif

  // Use supplied locale (ja_JP). This locale has a different alternate on some platforms.
#if defined(__APPLE__) || defined(_AIX)
  lcheck.template operator()<lfmt>(loc, SV("%d='00'\t%Od='00'\t%e=' 0'\t%Oe=' 0'\n"), 0d);
  lcheck.template operator()<lfmt>(loc, SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"), 1d);
  lcheck.template operator()<lfmt>(loc, SV("%d='31'\t%Od='31'\t%e='31'\t%Oe='31'\n"), 31d);
#  if defined(_AIX)
  check.template operator()<fmt>(SV("%d='55'\t%Od='55'\t%e='55'\t%Oe='55'\n"), 255d);
#  else
  check.template operator()<fmt>(SV("%d='255'\t%Od='255'\t%e='255'\t%Oe='255'\n"), 255d);
#  endif
#else  // defined(__APPLE__) || defined(_AIX)
  lcheck.template operator()<lfmt>(loc, SV("%d='00'\t%Od='〇'\t%e=' 0'\t%Oe='〇'\n"), 0d);
  lcheck.template operator()<lfmt>(loc, SV("%d='01'\t%Od='一'\t%e=' 1'\t%Oe='一'\n"), 1d);
  lcheck.template operator()<lfmt>(loc, SV("%d='31'\t%Od='三十一'\t%e='31'\t%Oe='三十一'\n"), 31d);
  lcheck.template operator()<lfmt>(loc, SV("%d='255'\t%Od='255'\t%e='255'\t%Oe='255'\n"), 255d);
#endif // defined(__APPLE__) || defined(_AIX)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test() {
  using namespace std::literals::chrono_literals;

  test_no_chrono_specs<CharT>();
  test_valid_values<CharT>();
  check_invalid_types<CharT>({SV("d"), SV("e"), SV("Od"), SV("Oe")}, 0d);

  check_exception("Expected '%' or '}' in the chrono format-string", SV("{:A"), 0d);
  check_exception("The chrono-specs contains a '{'", SV("{:%%{"), 0d);
  check_exception("End of input while parsing the modifier chrono conversion-spec", SV("{:%"), 0d);
  check_exception("End of input while parsing the modifier E", SV("{:%E"), 0d);
  check_exception("End of input while parsing the modifier O", SV("{:%O"), 0d);

  // Precision not allowed
  check_exception("Expected '%' or '}' in the chrono format-string", SV("{:.3}"), 0d);
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
