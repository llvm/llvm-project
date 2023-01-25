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
// UNSUPPORTED: msvc, target={{.+}}-windows-gnu

// TODO FMT It seems GCC uses too much memory in the CI and fails.
// UNSUPPORTED: gcc-12

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>
// class month_day;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month_day& md);

#include <chrono>
#include <cassert>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static std::basic_string<CharT> stream_c_locale(std::chrono::month_day md) {
  std::basic_stringstream<CharT> sstr;
  sstr << md;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::month_day md) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << md;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::month_day md) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << md;
  return sstr.str();
}

template <class CharT>
static void test() {
  using namespace std::literals::chrono_literals;

  assert(stream_c_locale<CharT>(std::chrono::month_day{std::chrono::month{0}, 0d}) ==
         SV("0 is not a valid month/00 is not a valid day"));
  assert(stream_c_locale<CharT>(std::chrono::month_day{std::chrono::month{0}, 1d}) == SV("0 is not a valid month/01"));
  assert(stream_c_locale<CharT>(std::chrono::month_day{std::chrono::month{1}, 255d}) ==
         SV("Jan/255 is not a valid day"));
  assert(stream_c_locale<CharT>(std::chrono::month_day{std::chrono::month{1}, 31d}) == SV("Jan/31"));
  // February is considered valid with 29 days; it lacks the year information
  // to do a proper validation.
  assert(stream_c_locale<CharT>(std::chrono::month_day{std::chrono::month{2}, 29d}) == SV("Feb/29"));
  // The month_day stream operator has no validation, this means never validate
  // dates don't get
  //   Jun/31 is not a valid month day
  // which is inconsistent with other stream operators.
  // TODO FMT file an issue about this.
  assert(stream_c_locale<CharT>(std::chrono::month_day{std::chrono::month{6}, 31d}) == SV("Jun/31"));

  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day{std::chrono::month{0}, 0d}) ==
         SV("0 is not a valid month/00 is not a valid day"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day{std::chrono::month{0}, 1d}) ==
         SV("0 is not a valid month/01"));
#if defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day{std::chrono::month{1}, 255d}) ==
         SV("jan/255 is not a valid day"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day{std::chrono::month{1}, 31d}) == SV("jan/31"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day{std::chrono::month{2}, 29d}) == SV("fév/29"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day{std::chrono::month{6}, 31d}) == SV("jui/31"));
#else  //  defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day{std::chrono::month{1}, 255d}) ==
         SV("janv./255 is not a valid day"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day{std::chrono::month{1}, 31d}) == SV("janv./31"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day{std::chrono::month{2}, 29d}) == SV("févr./29"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day{std::chrono::month{6}, 31d}) == SV("juin/31"));
#endif //  defined(__APPLE__)

  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{0}, 0d}) ==
         SV("0 is not a valid month/00 is not a valid day"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{0}, 1d}) ==
         SV("0 is not a valid month/01"));
#if defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{1}, 255d}) ==
         SV(" 1/255 is not a valid day"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{1}, 31d}) == SV(" 1/31"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{2}, 29d}) == SV(" 2/29"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{6}, 31d}) == SV(" 6/31"));
#elif defined(_AIX) || defined(_WIN32) //  defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{1}, 255d}) ==
         SV("1月/255 is not a valid day"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{1}, 31d}) == SV("1月/31"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{2}, 29d}) == SV("2月/29"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{6}, 31d}) == SV("6月/31"));
#else                                  //  defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{1}, 255d}) ==
         SV(" 1月/255 is not a valid day"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{1}, 31d}) == SV(" 1月/31"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{2}, 29d}) == SV(" 2月/29"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day{std::chrono::month{6}, 31d}) == SV(" 6月/31"));
#endif                                 //  defined(__APPLE__)
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
