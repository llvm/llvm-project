//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT Investigate Windows issues.
// UNSUPPORTED msvc, target={{.+}}-windows-gnu

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>
// class day;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const day& d);

#include <chrono>
#include <cassert>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static std::basic_string<CharT> stream_c_locale(std::chrono::day day) {
  std::basic_stringstream<CharT> sstr;
  sstr << day;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::day day) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << day;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::day day) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << day;
  return sstr.str();
}

template <class CharT>
static void test() {
  using namespace std::literals::chrono_literals;

  assert(stream_c_locale<CharT>(0d) == SV("00 is not a valid day"));
  assert(stream_c_locale<CharT>(1d) == SV("01"));
  assert(stream_c_locale<CharT>(31d) == SV("31"));
  assert(stream_c_locale<CharT>(255d) == SV("255 is not a valid day"));

  assert(stream_fr_FR_locale<CharT>(0d) == SV("00 is not a valid day"));
  assert(stream_fr_FR_locale<CharT>(1d) == SV("01"));
  assert(stream_fr_FR_locale<CharT>(31d) == SV("31"));
  assert(stream_fr_FR_locale<CharT>(255d) == SV("255 is not a valid day"));

  assert(stream_ja_JP_locale<CharT>(0d) == SV("00 is not a valid day"));
  assert(stream_ja_JP_locale<CharT>(1d) == SV("01"));
  assert(stream_ja_JP_locale<CharT>(31d) == SV("31"));
  assert(stream_ja_JP_locale<CharT>(255d) == SV("255 is not a valid day"));
}

int main(int, char**) {
  using namespace std::literals::chrono_literals;

  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
