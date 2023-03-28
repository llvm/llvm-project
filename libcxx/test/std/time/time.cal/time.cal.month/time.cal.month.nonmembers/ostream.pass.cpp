//===----------------------------------------------------------------------===//
//
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

// class month;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month& month);

#include <chrono>
#include <cassert>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static std::basic_string<CharT> stream_c_locale(std::chrono::month month) {
  std::basic_stringstream<CharT> sstr;
  sstr << month;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::month month) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << month;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::month month) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << month;
  return sstr.str();
}

template <class CharT>
static void test() {
  assert(stream_c_locale<CharT>(std::chrono::month{0}) == SV("0 is not a valid month"));
  assert(stream_c_locale<CharT>(std::chrono::month{1}) == SV("Jan"));
  assert(stream_c_locale<CharT>(std::chrono::month{2}) == SV("Feb"));
  assert(stream_c_locale<CharT>(std::chrono::month{3}) == SV("Mar"));
  assert(stream_c_locale<CharT>(std::chrono::month{4}) == SV("Apr"));
  assert(stream_c_locale<CharT>(std::chrono::month{5}) == SV("May"));
  assert(stream_c_locale<CharT>(std::chrono::month{6}) == SV("Jun"));
  assert(stream_c_locale<CharT>(std::chrono::month{7}) == SV("Jul"));
  assert(stream_c_locale<CharT>(std::chrono::month{8}) == SV("Aug"));
  assert(stream_c_locale<CharT>(std::chrono::month{9}) == SV("Sep"));
  assert(stream_c_locale<CharT>(std::chrono::month{10}) == SV("Oct"));
  assert(stream_c_locale<CharT>(std::chrono::month{11}) == SV("Nov"));
  assert(stream_c_locale<CharT>(std::chrono::month{12}) == SV("Dec"));
  assert(stream_c_locale<CharT>(std::chrono::month{13}) == SV("13 is not a valid month"));
  assert(stream_c_locale<CharT>(std::chrono::month{255}) == SV("255 is not a valid month"));

  assert(stream_fr_FR_locale<CharT>(std::chrono::month{0}) == SV("0 is not a valid month"));
#if defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{1}) == SV("jan"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{2}) == SV("fév"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{3}) == SV("mar"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{4}) == SV("avr"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{5}) == SV("mai"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{6}) == SV("jui"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{7}) == SV("jul"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{8}) == SV("aoû"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{9}) == SV("sep"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{10}) == SV("oct"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{11}) == SV("nov"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{12}) == SV("déc"));
#else //  defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{1}) == SV("janv."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{2}) == SV("févr."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{3}) == SV("mars"));
#  if defined(_WIN32) || defined(_AIX)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{4}) == SV("avr."));
#  else
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{4}) == SV("avril"));
#  endif
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{5}) == SV("mai"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{6}) == SV("juin"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{7}) == SV("juil."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{8}) == SV("août"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{9}) == SV("sept."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{10}) == SV("oct."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{11}) == SV("nov."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{12}) == SV("déc."));
#endif //  defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{13}) == SV("13 is not a valid month"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month{255}) == SV("255 is not a valid month"));

  assert(stream_ja_JP_locale<CharT>(std::chrono::month{0}) == SV("0 is not a valid month"));
#if defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{1}) == SV(" 1"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{2}) == SV(" 2"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{3}) == SV(" 3"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{4}) == SV(" 4"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{5}) == SV(" 5"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{6}) == SV(" 6"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{7}) == SV(" 7"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{8}) == SV(" 8"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{9}) == SV(" 9"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{10}) == SV("10"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{11}) == SV("11"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{12}) == SV("12"));
#else // defined(__APPLE__)
#  if defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{1}) == SV("1月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{2}) == SV("2月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{3}) == SV("3月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{4}) == SV("4月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{5}) == SV("5月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{6}) == SV("6月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{7}) == SV("7月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{8}) == SV("8月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{9}) == SV("9月"));
#  else  // defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{1}) == SV(" 1月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{2}) == SV(" 2月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{3}) == SV(" 3月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{4}) == SV(" 4月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{5}) == SV(" 5月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{6}) == SV(" 6月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{7}) == SV(" 7月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{8}) == SV(" 8月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{9}) == SV(" 9月"));
#  endif // defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{10}) == SV("10月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{11}) == SV("11月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{12}) == SV("12月"));
#endif   // defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{13}) == SV("13 is not a valid month"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month{255}) == SV("255 is not a valid month"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
