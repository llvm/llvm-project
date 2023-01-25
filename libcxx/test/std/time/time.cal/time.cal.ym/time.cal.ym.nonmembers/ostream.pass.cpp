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

// TODO FMT It seems GCC uses too much memory in the CI and fails.
// UNSUPPORTED: gcc-12

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// class year_month;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const year_month& ym);

#include <chrono>
#include <cassert>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static std::basic_string<CharT> stream_c_locale(std::chrono::year_month ym) {
  std::basic_stringstream<CharT> sstr;
  sstr << ym;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::year_month ym) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << ym;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::year_month ym) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << ym;
  return sstr.str();
}

template <class CharT>
static void test() {
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'768}, std::chrono::month{0}}) ==
         SV("-32768 is not a valid year/0 is not a valid month"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'768}, std::chrono::month{1}}) ==
         SV("-32768 is not a valid year/Jan"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'767}, std::chrono::month{2}}) ==
         SV("-32767/Feb"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{3}}) ==
         SV("0000/Mar"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{4}}) ==
         SV("1970/Apr"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{32'767}, std::chrono::month{5}}) ==
         SV("32767/May"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{6}}) ==
         SV("0000/Jun"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{7}}) ==
         SV("0000/Jul"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{8}}) ==
         SV("0000/Aug"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{9}}) ==
         SV("0000/Sep"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{10}}) ==
         SV("0000/Oct"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{11}}) ==
         SV("0000/Nov"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{12}}) ==
         SV("0000/Dec"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{13}}) ==
         SV("0000/13 is not a valid month"));
  assert(stream_c_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'768}, std::chrono::month{255}}) ==
         SV("-32768 is not a valid year/255 is not a valid month"));

  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'768}, std::chrono::month{0}}) ==
         SV("-32768 is not a valid year/0 is not a valid month"));
#if defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'768}, std::chrono::month{1}}) ==
         SV("-32768 is not a valid year/jan"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'767}, std::chrono::month{2}}) ==
         SV("-32767/fév"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{3}}) ==
         SV("0000/mar"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{4}}) ==
         SV("1970/avr"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{32'767}, std::chrono::month{5}}) ==
         SV("32767/mai"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{6}}) ==
         SV("0000/jui"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{7}}) ==
         SV("0000/jul"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{8}}) ==
         SV("0000/aoû"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{9}}) ==
         SV("0000/sep"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{10}}) ==
         SV("0000/oct"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{11}}) ==
         SV("0000/nov"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{12}}) ==
         SV("0000/déc"));
#else    //  defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'768}, std::chrono::month{1}}) ==
         SV("-32768 is not a valid year/janv."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'767}, std::chrono::month{2}}) ==
         SV("-32767/févr."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{3}}) ==
         SV("0000/mars"));
#  if defined(_WIN32) || defined(_AIX)
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{4}}) ==
         SV("1970/avr."));
#  else  // defined(_WIN32) || defined(_AIX)
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{4}}) ==
         SV("1970/avril"));
#  endif // defined(_WIN32) || defined(_AIX)
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{32'767}, std::chrono::month{5}}) ==
         SV("32767/mai"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{6}}) ==
         SV("0000/juin"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{7}}) ==
         SV("0000/juil."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{8}}) ==
         SV("0000/août"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{9}}) ==
         SV("0000/sept."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{10}}) ==
         SV("0000/oct."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{11}}) ==
         SV("0000/nov."));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{12}}) ==
         SV("0000/déc."));
#endif   //  defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{13}}) ==
         SV("0000/13 is not a valid month"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'768}, std::chrono::month{255}}) ==
         SV("-32768 is not a valid year/255 is not a valid month"));

  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'768}, std::chrono::month{0}}) ==
         SV("-32768 is not a valid year/0 is not a valid month"));
#if defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'768}, std::chrono::month{1}}) ==
         SV("-32768 is not a valid year/ 1"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'767}, std::chrono::month{2}}) ==
         SV("-32767/ 2"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{3}}) ==
         SV("0000/ 3"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{4}}) ==
         SV("1970/ 4"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{32'767}, std::chrono::month{5}}) ==
         SV("32767/ 5"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{6}}) ==
         SV("0000/ 6"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{7}}) ==
         SV("0000/ 7"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{8}}) ==
         SV("0000/ 8"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{9}}) ==
         SV("0000/ 9"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{10}}) ==
         SV("0000/10"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{11}}) ==
         SV("0000/11"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{12}}) ==
         SV("0000/12"));
#else    // defined(__APPLE__)
#  if defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'768}, std::chrono::month{1}}) ==
         SV("-32768 is not a valid year/1月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'767}, std::chrono::month{2}}) ==
         SV("-32767/2月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{3}}) ==
         SV("0000/3月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{4}}) ==
         SV("1970/4月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{32'767}, std::chrono::month{5}}) ==
         SV("32767/5月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{6}}) ==
         SV("0000/6月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{7}}) ==
         SV("0000/7月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{8}}) ==
         SV("0000/8月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{9}}) ==
         SV("0000/9月"));
#  else  // defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'768}, std::chrono::month{1}}) ==
         SV("-32768 is not a valid year/ 1月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'767}, std::chrono::month{2}}) ==
         SV("-32767/ 2月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{3}}) ==
         SV("0000/ 3月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{1970}, std::chrono::month{4}}) ==
         SV("1970/ 4月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{32'767}, std::chrono::month{5}}) ==
         SV("32767/ 5月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{6}}) ==
         SV("0000/ 6月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{7}}) ==
         SV("0000/ 7月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{8}}) ==
         SV("0000/ 8月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{9}}) ==
         SV("0000/ 9月"));
#  endif // defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{10}}) ==
         SV("0000/10月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{11}}) ==
         SV("0000/11月"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{12}}) ==
         SV("0000/12月"));
#endif   // defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{0}, std::chrono::month{13}}) ==
         SV("0000/13 is not a valid month"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month{std::chrono::year{-32'768}, std::chrono::month{255}}) ==
         SV("-32768 is not a valid year/255 is not a valid month"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
