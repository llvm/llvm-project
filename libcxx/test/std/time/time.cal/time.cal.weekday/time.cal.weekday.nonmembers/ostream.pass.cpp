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
// class weekday;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const weekday& wd);

#include <chrono>
#include <cassert>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static std::basic_string<CharT> stream_c_locale(std::chrono::weekday weekday) {
  std::basic_stringstream<CharT> sstr;
  sstr << weekday;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::weekday weekday) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << weekday;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::weekday weekday) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << weekday;
  return sstr.str();
}

template <class CharT>
static void test() {
  assert(stream_c_locale<CharT>(std::chrono::weekday(0)) == SV("Sun"));
  assert(stream_c_locale<CharT>(std::chrono::weekday(1)) == SV("Mon"));
  assert(stream_c_locale<CharT>(std::chrono::weekday(2)) == SV("Tue"));
  assert(stream_c_locale<CharT>(std::chrono::weekday(3)) == SV("Wed"));
  assert(stream_c_locale<CharT>(std::chrono::weekday(4)) == SV("Thu"));
  assert(stream_c_locale<CharT>(std::chrono::weekday(5)) == SV("Fri"));
  assert(stream_c_locale<CharT>(std::chrono::weekday(6)) == SV("Sat"));
  assert(stream_c_locale<CharT>(std::chrono::weekday(7)) == SV("Sun"));
  assert(stream_c_locale<CharT>(std::chrono::weekday(8)) == SV("8 is not a valid weekday"));
  assert(stream_c_locale<CharT>(std::chrono::weekday(255)) == SV("255 is not a valid weekday"));

#if defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(0)) == SV("Dim"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(1)) == SV("Lun"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(2)) == SV("Mar"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(3)) == SV("Mer"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(4)) == SV("Jeu"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(5)) == SV("Ven"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(6)) == SV("Sam"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(7)) == SV("Dim"));
#else
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(0)) == SV("dim."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(1)) == SV("lun."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(2)) == SV("mar."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(3)) == SV("mer."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(4)) == SV("jeu."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(5)) == SV("ven."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(6)) == SV("sam."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(7)) == SV("dim."));
#endif
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(8)) == SV("8 is not a valid weekday"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday(255)) == SV("255 is not a valid weekday"));

  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday(0)) == SV("日"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday(1)) == SV("月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday(2)) == SV("火"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday(3)) == SV("水"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday(4)) == SV("木"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday(5)) == SV("金"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday(6)) == SV("土"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday(7)) == SV("日"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday(8)) == SV("8 is not a valid weekday"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday(255)) == SV("255 is not a valid weekday"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
