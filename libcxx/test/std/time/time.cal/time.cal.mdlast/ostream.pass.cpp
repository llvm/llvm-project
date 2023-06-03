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
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// class month_day_last;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month_day_last& mdl);

#include <chrono>
#include <cassert>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

#define TEST_EQUAL(OUT, EXPECTED)                                                                                      \
  TEST_REQUIRE(OUT == EXPECTED,                                                                                        \
               TEST_WRITE_CONCATENATED(                                                                                \
                   "\nExpression      ", #OUT, "\nExpected output ", EXPECTED, "\nActual output   ", OUT, '\n'));

template <class CharT>
static std::basic_string<CharT> stream_c_locale(std::chrono::month_day_last mdl) {
  std::basic_stringstream<CharT> sstr;
  sstr << mdl;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::month_day_last mdl) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << mdl;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::month_day_last mdl) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << mdl;
  return sstr.str();
}

template <class CharT>
static void test() {
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{0}}),
             SV("0 is not a valid month/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}), SV("Jan/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}), SV("Feb/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}), SV("Mar/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}), SV("Apr/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}), SV("May/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}), SV("Jun/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}), SV("Jul/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}), SV("Aug/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}), SV("Sep/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{10}}), SV("Oct/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{11}}), SV("Nov/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{12}}), SV("Dec/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{13}}),
             SV("13 is not a valid month/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{255}}),
             SV("255 is not a valid month/last"));

  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{0}}),
             SV("0 is not a valid month/last"));
#if defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}), SV("jan/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}), SV("fév/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}), SV("mar/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}), SV("avr/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}), SV("mai/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}), SV("jui/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}), SV("jul/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}), SV("aoû/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}), SV("sep/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{10}}), SV("oct/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{11}}), SV("nov/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{12}}), SV("déc/last"));
#else //  defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}), SV("janv./last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}), SV("févr./last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}), SV("mars/last"));
#  if defined(_WIN32) || defined(_AIX)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}), SV("avr./last"));
#  else
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}), SV("avril/last"));
#  endif
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}), SV("mai/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}), SV("juin/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}), SV("juil./last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}), SV("août/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}), SV("sept./last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{10}}), SV("oct./last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{11}}), SV("nov./last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{12}}), SV("déc./last"));
#endif //  defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{13}}),
             SV("13 is not a valid month/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{255}}),
             SV("255 is not a valid month/last"));

  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{0}}),
             SV("0 is not a valid month/last"));
#if defined(__APPLE__) || defined(_WIN32)
#  if defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}), SV(" 1/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}), SV(" 2/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}), SV(" 3/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}), SV(" 4/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}), SV(" 5/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}), SV(" 6/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}), SV(" 7/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}), SV(" 8/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}), SV(" 9/last"));
#  else  // defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}), SV("1/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}), SV("2/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}), SV("3/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}), SV("4/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}), SV("5/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}), SV("6/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}), SV("7/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}), SV("8/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}), SV("9/last"));
#  endif // defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{10}}), SV("10/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{11}}), SV("11/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{12}}), SV("12/last"));
#else // defined(__APPLE__) || defined(_WIN32)
#  if defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}), SV("1月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}), SV("2月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}), SV("3月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}), SV("4月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}), SV("5月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}), SV("6月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}), SV("7月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}), SV("8月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}), SV("9月/last"));
#  else  // defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}), SV(" 1月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}), SV(" 2月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}), SV(" 3月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}), SV(" 4月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}), SV(" 5月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}), SV(" 6月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}), SV(" 7月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}), SV(" 8月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}), SV(" 9月/last"));
#  endif // defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{10}}), SV("10月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{11}}), SV("11月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{12}}), SV("12月/last"));
#endif   // defined(__APPLE__) || defined(_WIN32)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{13}}),
             SV("13 is not a valid month/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{255}}),
             SV("255 is not a valid month/last"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
