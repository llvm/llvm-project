//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO(mordante) Investigate
// UNSUPPORTED: apple-clang

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// ADDITIONAL_COMPILE_FLAGS: -DFR_THOU_SEP=%{LOCALE_CONV_FR_FR_UTF_8_THOUSANDS_SEP}
// ADDITIONAL_COMPILE_FLAGS: -DFR_DEC_POINT=%{LOCALE_CONV_FR_FR_UTF_8_DECIMAL_POINT}

// <chrono>

// template<class Rep, class Period = ratio<1>> class duration;

// template<class charT, class traits, class Rep, class Period>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os,
//                const duration<Rep, Period>& d);

#include <chrono>

#include <cassert>
#include <concepts>
#include <ratio>
#include <sstream>

#include "make_string.h"
#include "locale_helpers.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT, class Rep, class Period>
static std::basic_string<CharT> stream_c_locale(std::chrono::duration<Rep, Period> duration) {
  std::basic_stringstream<CharT> sstr;
  sstr.precision(4);
  sstr << std::fixed << duration;
  return sstr.str();
}

template <class CharT, class Rep, class Period>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::duration<Rep, Period> duration) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr.precision(4);
  sstr << std::fixed << duration;
  return sstr.str();
}

template <class CharT, class Rep, class Period>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::duration<Rep, Period> duration) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr.precision(4);
  sstr << std::fixed << duration;
  return sstr.str();
}

template <class CharT>
static void test_values() {
  using namespace std::literals::chrono_literals;

  assert(stream_c_locale<CharT>(-1'000'000s) == SV("-1000000s"));
  assert(stream_c_locale<CharT>(1'000'000s) == SV("1000000s"));
  assert(stream_c_locale<CharT>(-1'000.123456s) == SV("-1000.1235s"));
  assert(stream_c_locale<CharT>(1'000.123456s) == SV("1000.1235s"));

  if constexpr (std::same_as<CharT, char>) {
#if defined(__APPLE__)
    assert(stream_fr_FR_locale<CharT>(-1'000'000s) == SV("-1000000s"));
    assert(stream_fr_FR_locale<CharT>(1'000'000s) == SV("1000000s"));
    assert(stream_fr_FR_locale<CharT>(-1'000.123456s) == SV("-1000,1235s"));
    assert(stream_fr_FR_locale<CharT>(1'000.123456s) == SV("1000,1235s"));
#else
    assert(stream_fr_FR_locale<CharT>(-1'000'000s) == SV("-1 000 000s"));
    assert(stream_fr_FR_locale<CharT>(1'000'000s) == SV("1 000 000s"));
    assert(stream_fr_FR_locale<CharT>(-1'000.123456s) == SV("-1 000,1235s"));
    assert(stream_fr_FR_locale<CharT>(1'000.123456s) == SV("1 000,1235s"));
#endif
  } else {
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    assert(stream_fr_FR_locale<CharT>(-1'000'000s) == L"-1" FR_THOU_SEP "000" FR_THOU_SEP "000s");
    assert(stream_fr_FR_locale<CharT>(1'000'000s) == L"1" FR_THOU_SEP "000" FR_THOU_SEP "000s");
    assert(stream_fr_FR_locale<CharT>(-1'000.123456s) == L"-1" FR_THOU_SEP "000" FR_DEC_POINT "1235s");
    assert(stream_fr_FR_locale<CharT>(1'000.123456s) == L"1" FR_THOU_SEP "000" FR_DEC_POINT "1235s");
#endif
  }

  assert(stream_ja_JP_locale<CharT>(-1'000'000s) == SV("-1,000,000s"));
  assert(stream_ja_JP_locale<CharT>(1'000'000s) == SV("1,000,000s"));
  assert(stream_ja_JP_locale<CharT>(-1'000.123456s) == SV("-1,000.1235s"));
  assert(stream_ja_JP_locale<CharT>(1'000.123456s) == SV("1,000.1235s"));
}

template <class CharT>
static void test_units() {
  using namespace std::literals::chrono_literals;

  // C locale
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::atto>(0)) == SV("0as"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::femto>(0)) == SV("0fs"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::pico>(0)) == SV("0ps"));
  assert(stream_c_locale<CharT>(0ns) == SV("0ns"));
#ifndef TEST_HAS_NO_UNICODE
  assert(stream_c_locale<CharT>(0us) == SV("0\u00b5s"));
#else
  assert(stream_c_locale<CharT>(0us) == SV("0us"));
#endif
  assert(stream_c_locale<CharT>(0ms) == SV("0ms"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::centi>(0)) == SV("0cs"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::deci>(0)) == SV("0ds"));

  assert(stream_c_locale<CharT>(0s) == SV("0s"));

  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::deca>(0)) == SV("0das"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::hecto>(0)) == SV("0hs"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::kilo>(0)) == SV("0ks"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::mega>(0)) == SV("0Ms"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::giga>(0)) == SV("0Gs"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::tera>(0)) == SV("0Ts"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::peta>(0)) == SV("0Ps"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::exa>(0)) == SV("0Es"));

  assert(stream_c_locale<CharT>(0min) == SV("0min"));
  assert(stream_c_locale<CharT>(0h) == SV("0h"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::ratio<86400>>(0)) == SV("0d"));

  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::ratio<42>>(0)) == SV("0[42]s"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::ratio<33, 3>>(0)) == SV("0[11]s"));
  assert(stream_c_locale<CharT>(std::chrono::duration<int, std::ratio<11, 9>>(0)) == SV("0[11/9]s"));

  // fr_FR locale
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::atto>(0)) == SV("0as"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::femto>(0)) == SV("0fs"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::pico>(0)) == SV("0ps"));
  assert(stream_fr_FR_locale<CharT>(0ns) == SV("0ns"));
#ifndef TEST_HAS_NO_UNICODE
  assert(stream_fr_FR_locale<CharT>(0us) == SV("0\u00b5s"));
#else
  assert(stream_fr_FR_locale<CharT>(0us) == SV("0us"));
#endif
  assert(stream_fr_FR_locale<CharT>(0ms) == SV("0ms"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::centi>(0)) == SV("0cs"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::deci>(0)) == SV("0ds"));

  assert(stream_fr_FR_locale<CharT>(0s) == SV("0s"));

  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::deca>(0)) == SV("0das"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::hecto>(0)) == SV("0hs"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::kilo>(0)) == SV("0ks"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::mega>(0)) == SV("0Ms"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::giga>(0)) == SV("0Gs"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::tera>(0)) == SV("0Ts"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::peta>(0)) == SV("0Ps"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::exa>(0)) == SV("0Es"));

  assert(stream_fr_FR_locale<CharT>(0min) == SV("0min"));
  assert(stream_fr_FR_locale<CharT>(0h) == SV("0h"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::ratio<86400>>(0)) == SV("0d"));

  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::ratio<42>>(0)) == SV("0[42]s"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::ratio<33, 3>>(0)) == SV("0[11]s"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<int, std::ratio<11, 9>>(0)) == SV("0[11/9]s"));

  // ja_JP locale
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::atto>(0)) == SV("0as"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::femto>(0)) == SV("0fs"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::pico>(0)) == SV("0ps"));
  assert(stream_ja_JP_locale<CharT>(0ns) == SV("0ns"));
#ifndef TEST_HAS_NO_UNICODE
  assert(stream_ja_JP_locale<CharT>(0us) == SV("0\u00b5s"));
#else
  assert(stream_ja_JP_locale<CharT>(0us) == SV("0us"));
#endif
  assert(stream_ja_JP_locale<CharT>(0ms) == SV("0ms"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::centi>(0)) == SV("0cs"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::deci>(0)) == SV("0ds"));

  assert(stream_ja_JP_locale<CharT>(0s) == SV("0s"));

  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::deca>(0)) == SV("0das"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::hecto>(0)) == SV("0hs"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::kilo>(0)) == SV("0ks"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::mega>(0)) == SV("0Ms"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::giga>(0)) == SV("0Gs"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::tera>(0)) == SV("0Ts"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::peta>(0)) == SV("0Ps"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::exa>(0)) == SV("0Es"));

  assert(stream_ja_JP_locale<CharT>(0min) == SV("0min"));
  assert(stream_ja_JP_locale<CharT>(0h) == SV("0h"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::ratio<86400>>(0)) == SV("0d"));

  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::ratio<42>>(0)) == SV("0[42]s"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::ratio<33, 3>>(0)) == SV("0[11]s"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<int, std::ratio<11, 9>>(0)) == SV("0[11/9]s"));
}

template <class CharT>
static void test_unsigned_types() {
  // Reported in https://llvm.org/PR96820
  using namespace std::literals::chrono_literals;

  // C locale
  assert(stream_c_locale<CharT>(std::chrono::duration<unsigned short, std::atto>(0)) == SV("0as"));
  assert(stream_c_locale<CharT>(std::chrono::duration<unsigned, std::femto>(0)) == SV("0fs"));
  assert(stream_c_locale<CharT>(std::chrono::duration<unsigned long, std::pico>(0)) == SV("0ps"));
  assert(stream_c_locale<CharT>(std::chrono::duration<unsigned long long, std::nano>(0)) == SV("0ns"));

  // fr_FR locale
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<unsigned short, std::atto>(0)) == SV("0as"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<unsigned, std::femto>(0)) == SV("0fs"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<unsigned long, std::pico>(0)) == SV("0ps"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::duration<unsigned long long, std::nano>(0)) == SV("0ns"));

  // ja_JP locale
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<unsigned short, std::atto>(0)) == SV("0as"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<unsigned, std::femto>(0)) == SV("0fs"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<unsigned long, std::pico>(0)) == SV("0ps"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::duration<unsigned long long, std::nano>(0)) == SV("0ns"));
}

template <class CharT>
static void test() {
  test_values<CharT>();
  test_units<CharT>();
  test_unsigned_types<CharT>();
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
