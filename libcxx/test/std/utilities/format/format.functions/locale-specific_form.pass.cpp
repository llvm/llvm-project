//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT Evaluate gcc-12 status
// UNSUPPORTED: gcc-12

// REQUIRES: locale.en_US.UTF-8

// <format>

// This test the locale-specific form for these formatting functions:
//
//  // [format.functions], formatting functions
//  template<class... Args>
//    string format(string_view fmt, const Args&... args);
//  template<class... Args>
//    wstring format(wstring_view fmt, const Args&... args);
//  template<class... Args>
//    string format(const locale& loc, string_view fmt, const Args&... args);
//  template<class... Args>
//    wstring format(const locale& loc, wstring_view fmt, const Args&... args);
//
//  string vformat(string_view fmt, format_args args);
//  wstring vformat(wstring_view fmt, wformat_args args);
//  string vformat(const locale& loc, string_view fmt, format_args args);
//  wstring vformat(const locale& loc, wstring_view fmt, wformat_args args);
//
//  template<class Out, class... Args>
//    Out format_to(Out out, string_view fmt, const Args&... args);
//  template<class Out, class... Args>
//    Out format_to(Out out, wstring_view fmt, const Args&... args);
//  template<class Out, class... Args>
//    Out format_to(Out out, const locale& loc, string_view fmt, const Args&... args);
//  template<class Out, class... Args>
//    Out format_to(Out out, const locale& loc, wstring_view fmt, const Args&... args);
//
//  template<class Out>
//    Out vformat_to(Out out, string_view fmt, format_args args);
//  template<class Out>
//    Out vformat_to(Out out, wstring_view fmt, wformat_args args);
//  template<class Out>
//    Out vformat_to(Out out, const locale& loc, string_view fmt,
//                   format_args args);
//  template<class Out>
//    Out vformat_to(Out out, const locale& loc, wstring_view fmt,
//                   wformat_arg args);
//
//  template<class Out> struct format_to_n_result {
//    Out out;
//    iter_difference_t<Out> size;
//  };
//
// template<class Out, class... Args>
//    format_to_n_result<Out> format_to_n(Out out, iter_difference_t<Out> n,
//                                        string_view fmt, const Args&... args);
//  template<class Out, class... Args>
//    format_to_n_result<Out> format_to_n(Out out, iter_difference_t<Out> n,
//                                        wstring_view fmt, const Args&... args);
//  template<class Out, class... Args>
//    format_to_n_result<Out> format_to_n(Out out, iter_difference_t<Out> n,
//                                        const locale& loc, string_view fmt,
//                                        const Args&... args);
//  template<class Out, class... Args>
//    format_to_n_result<Out> format_to_n(Out out, iter_difference_t<Out> n,
//                                        const locale& loc, wstring_view fmt,
//                                        const Args&... args);
//
//  template<class... Args>
//    size_t formatted_size(string_view fmt, const Args&... args);
//  template<class... Args>
//    size_t formatted_size(wstring_view fmt, const Args&... args);
//  template<class... Args>
//    size_t formatted_size(const locale& loc, string_view fmt, const Args&... args);
//  template<class... Args>
//    size_t formatted_size(const locale& loc, wstring_view fmt, const Args&... args);

#include <format>
#include <cassert>
#include <iostream>
#include <vector>

#include "test_macros.h"
#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "format_tests.h"
#include "string_literal.h"
#include "test_format_string.h"

#define STR(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
struct numpunct;

template <>
struct numpunct<char> : std::numpunct<char> {
  string_type do_truename() const override { return "yes"; }
  string_type do_falsename() const override { return "no"; }

  std::string do_grouping() const override { return "\1\2\3\2\1"; };
  char do_thousands_sep() const override { return '_'; }
  char do_decimal_point() const override { return '#'; }
};

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
template <>
struct numpunct<wchar_t> : std::numpunct<wchar_t> {
  string_type do_truename() const override { return L"yes"; }
  string_type do_falsename() const override { return L"no"; }

  std::string do_grouping() const override { return "\1\2\3\2\1"; };
  wchar_t do_thousands_sep() const override { return L'_'; }
  wchar_t do_decimal_point() const override { return L'#'; }
};
#endif

template <class CharT, class... Args>
void test(std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) {
  // *** format ***
  {
    std::basic_string<CharT> out = std::format(fmt, std::forward<Args>(args)...);
    if constexpr (std::same_as<CharT, char>)
      if (out != expected)
        std::cerr << "\nFormat string   " << fmt.get() << "\nExpected output " << expected << "\nActual output   "
                  << out << '\n';
    assert(out == expected);
  }
  // *** vformat ***
  {
    std::basic_string<CharT> out = std::vformat(fmt.get(), std::make_format_args<context_t<CharT>>(args...));
    assert(out == expected);
  }
  // *** format_to ***
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::format_to(out.begin(), fmt, std::forward<Args>(args)...);
    assert(it == out.end());
    assert(out == expected);
  }
  // *** vformat_to ***
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::vformat_to(out.begin(), fmt.get(), std::make_format_args<context_t<CharT>>(args...));
    assert(it == out.end());
    assert(out == expected);
  }
  // *** format_to_n ***
  {
    std::basic_string<CharT> out;
    std::format_to_n_result result = std::format_to_n(std::back_inserter(out), 1000, fmt, std::forward<Args>(args)...);
    using diff_type = decltype(result.size);
    diff_type formatted_size       = std::formatted_size(fmt, std::forward<Args>(args)...);
    diff_type size = std::min<diff_type>(1000, formatted_size);

    assert(result.size == formatted_size);
    assert(out == expected.substr(0, size));
  }
  // *** formatted_size ***
  {
    size_t size = std::formatted_size(fmt, std::forward<Args>(args)...);
    assert(size == expected.size());
  }
}

template <class CharT, class... Args>
void test(
    std::basic_string_view<CharT> expected, std::locale loc, test_format_string<CharT, Args...> fmt, Args&&... args) {
  // *** format ***
  {
    std::basic_string<CharT> out = std::format(loc, fmt, std::forward<Args>(args)...);
    if constexpr (std::same_as<CharT, char>)
      if (out != expected)
        std::cerr << "\nFormat string   " << fmt.get() << "\nExpected output " << expected << "\nActual output   "
                  << out << '\n';
    assert(out == expected);
  }
  // *** vformat ***
  {
    std::basic_string<CharT> out = std::vformat(loc, fmt.get(), std::make_format_args<context_t<CharT>>(args...));
    assert(out == expected);
  }
  // *** format_to ***
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::format_to(out.begin(), loc, fmt, std::forward<Args>(args)...);
    assert(it == out.end());
    assert(out == expected);
  }
  // *** vformat_to ***
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::vformat_to(out.begin(), loc, fmt.get(), std::make_format_args<context_t<CharT>>(args...));
    assert(it == out.end());
    assert(out == expected);
  }
  // *** format_to_n ***
  {
    std::basic_string<CharT> out;
    std::format_to_n_result result =
        std::format_to_n(std::back_inserter(out), 1000, loc, fmt, std::forward<Args>(args)...);
    using diff_type = decltype(result.size);
    diff_type formatted_size = std::formatted_size(loc, fmt, std::forward<Args>(args)...);
    diff_type size = std::min<diff_type>(1000, formatted_size);

    assert(result.size == formatted_size);
    assert(out == expected.substr(0, size));
  }
  // *** formatted_size ***
  {
    size_t size = std::formatted_size(loc, fmt, std::forward<Args>(args)...);
    assert(size == expected.size());
  }
}

#ifndef TEST_HAS_NO_UNICODE
template <class CharT>
struct numpunct_unicode;

template <>
struct numpunct_unicode<char> : std::numpunct<char> {
  string_type do_truename() const override { return "gültig"; }
  string_type do_falsename() const override { return "ungültig"; }
};

#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
template <>
struct numpunct_unicode<wchar_t> : std::numpunct<wchar_t> {
  string_type do_truename() const override { return L"gültig"; }
  string_type do_falsename() const override { return L"ungültig"; }
};
#  endif
#endif // TEST_HAS_NO_UNICODE

template <class CharT>
void test_bool() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());

  std::locale::global(std::locale(LOCALE_en_US_UTF_8));
  assert(std::locale().name() == LOCALE_en_US_UTF_8);
  test(SV("true"), SV("{:L}"), true);
  test(SV("false"), SV("{:L}"), false);

  test(SV("yes"), loc, SV("{:L}"), true);
  test(SV("no"), loc, SV("{:L}"), false);

  std::locale::global(loc);
  test(SV("yes"), SV("{:L}"), true);
  test(SV("no"), SV("{:L}"), false);

  test(SV("true"), std::locale(LOCALE_en_US_UTF_8), SV("{:L}"), true);
  test(SV("false"), std::locale(LOCALE_en_US_UTF_8), SV("{:L}"), false);

#ifndef TEST_HAS_NO_UNICODE
  std::locale loc_unicode = std::locale(std::locale(), new numpunct_unicode<CharT>());

  test(SV("gültig"), loc_unicode, SV("{:L}"), true);
  test(SV("ungültig"), loc_unicode, SV("{:L}"), false);

  test(SV("gültig   "), loc_unicode, SV("{:9L}"), true);
  test(SV("gültig!!!"), loc_unicode, SV("{:!<9L}"), true);
  test(SV("_gültig__"), loc_unicode, SV("{:_^9L}"), true);
  test(SV("   gültig"), loc_unicode, SV("{:>9L}"), true);
#endif // TEST_HAS_NO_UNICODE
}

template <class CharT>
void test_integer() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Decimal ***
  std::locale::global(en_US);
  test(SV("0"), SV("{:L}"), 0);
  test(SV("1"), SV("{:L}"), 1);
  test(SV("10"), SV("{:L}"), 10);
  test(SV("100"), SV("{:L}"), 100);
  test(SV("1,000"), SV("{:L}"), 1'000);
  test(SV("10,000"), SV("{:L}"), 10'000);
  test(SV("100,000"), SV("{:L}"), 100'000);
  test(SV("1,000,000"), SV("{:L}"), 1'000'000);
  test(SV("10,000,000"), SV("{:L}"), 10'000'000);
  test(SV("100,000,000"), SV("{:L}"), 100'000'000);
  test(SV("1,000,000,000"), SV("{:L}"), 1'000'000'000);

  test(SV("-1"), SV("{:L}"), -1);
  test(SV("-10"), SV("{:L}"), -10);
  test(SV("-100"), SV("{:L}"), -100);
  test(SV("-1,000"), SV("{:L}"), -1'000);
  test(SV("-10,000"), SV("{:L}"), -10'000);
  test(SV("-100,000"), SV("{:L}"), -100'000);
  test(SV("-1,000,000"), SV("{:L}"), -1'000'000);
  test(SV("-10,000,000"), SV("{:L}"), -10'000'000);
  test(SV("-100,000,000"), SV("{:L}"), -100'000'000);
  test(SV("-1,000,000,000"), SV("{:L}"), -1'000'000'000);

  std::locale::global(loc);
  test(SV("0"), SV("{:L}"), 0);
  test(SV("1"), SV("{:L}"), 1);
  test(SV("1_0"), SV("{:L}"), 10);
  test(SV("10_0"), SV("{:L}"), 100);
  test(SV("1_00_0"), SV("{:L}"), 1'000);
  test(SV("10_00_0"), SV("{:L}"), 10'000);
  test(SV("100_00_0"), SV("{:L}"), 100'000);
  test(SV("1_000_00_0"), SV("{:L}"), 1'000'000);
  test(SV("10_000_00_0"), SV("{:L}"), 10'000'000);
  test(SV("1_00_000_00_0"), SV("{:L}"), 100'000'000);
  test(SV("1_0_00_000_00_0"), SV("{:L}"), 1'000'000'000);

  test(SV("-1"), SV("{:L}"), -1);
  test(SV("-1_0"), SV("{:L}"), -10);
  test(SV("-10_0"), SV("{:L}"), -100);
  test(SV("-1_00_0"), SV("{:L}"), -1'000);
  test(SV("-10_00_0"), SV("{:L}"), -10'000);
  test(SV("-100_00_0"), SV("{:L}"), -100'000);
  test(SV("-1_000_00_0"), SV("{:L}"), -1'000'000);
  test(SV("-10_000_00_0"), SV("{:L}"), -10'000'000);
  test(SV("-1_00_000_00_0"), SV("{:L}"), -100'000'000);
  test(SV("-1_0_00_000_00_0"), SV("{:L}"), -1'000'000'000);

  test(SV("0"), en_US, SV("{:L}"), 0);
  test(SV("1"), en_US, SV("{:L}"), 1);
  test(SV("10"), en_US, SV("{:L}"), 10);
  test(SV("100"), en_US, SV("{:L}"), 100);
  test(SV("1,000"), en_US, SV("{:L}"), 1'000);
  test(SV("10,000"), en_US, SV("{:L}"), 10'000);
  test(SV("100,000"), en_US, SV("{:L}"), 100'000);
  test(SV("1,000,000"), en_US, SV("{:L}"), 1'000'000);
  test(SV("10,000,000"), en_US, SV("{:L}"), 10'000'000);
  test(SV("100,000,000"), en_US, SV("{:L}"), 100'000'000);
  test(SV("1,000,000,000"), en_US, SV("{:L}"), 1'000'000'000);

  test(SV("-1"), en_US, SV("{:L}"), -1);
  test(SV("-10"), en_US, SV("{:L}"), -10);
  test(SV("-100"), en_US, SV("{:L}"), -100);
  test(SV("-1,000"), en_US, SV("{:L}"), -1'000);
  test(SV("-10,000"), en_US, SV("{:L}"), -10'000);
  test(SV("-100,000"), en_US, SV("{:L}"), -100'000);
  test(SV("-1,000,000"), en_US, SV("{:L}"), -1'000'000);
  test(SV("-10,000,000"), en_US, SV("{:L}"), -10'000'000);
  test(SV("-100,000,000"), en_US, SV("{:L}"), -100'000'000);
  test(SV("-1,000,000,000"), en_US, SV("{:L}"), -1'000'000'000);

  std::locale::global(en_US);
  test(SV("0"), loc, SV("{:L}"), 0);
  test(SV("1"), loc, SV("{:L}"), 1);
  test(SV("1_0"), loc, SV("{:L}"), 10);
  test(SV("10_0"), loc, SV("{:L}"), 100);
  test(SV("1_00_0"), loc, SV("{:L}"), 1'000);
  test(SV("10_00_0"), loc, SV("{:L}"), 10'000);
  test(SV("100_00_0"), loc, SV("{:L}"), 100'000);
  test(SV("1_000_00_0"), loc, SV("{:L}"), 1'000'000);
  test(SV("10_000_00_0"), loc, SV("{:L}"), 10'000'000);
  test(SV("1_00_000_00_0"), loc, SV("{:L}"), 100'000'000);
  test(SV("1_0_00_000_00_0"), loc, SV("{:L}"), 1'000'000'000);

  test(SV("-1"), loc, SV("{:L}"), -1);
  test(SV("-1_0"), loc, SV("{:L}"), -10);
  test(SV("-10_0"), loc, SV("{:L}"), -100);
  test(SV("-1_00_0"), loc, SV("{:L}"), -1'000);
  test(SV("-10_00_0"), loc, SV("{:L}"), -10'000);
  test(SV("-100_00_0"), loc, SV("{:L}"), -100'000);
  test(SV("-1_000_00_0"), loc, SV("{:L}"), -1'000'000);
  test(SV("-10_000_00_0"), loc, SV("{:L}"), -10'000'000);
  test(SV("-1_00_000_00_0"), loc, SV("{:L}"), -100'000'000);
  test(SV("-1_0_00_000_00_0"), loc, SV("{:L}"), -1'000'000'000);

  // *** Binary ***
  std::locale::global(en_US);
  test(SV("0"), SV("{:Lb}"), 0b0);
  test(SV("1"), SV("{:Lb}"), 0b1);
  test(SV("1,000,000,000"), SV("{:Lb}"), 0b1'000'000'000);

  test(SV("0b0"), SV("{:#Lb}"), 0b0);
  test(SV("0b1"), SV("{:#Lb}"), 0b1);
  test(SV("0b1,000,000,000"), SV("{:#Lb}"), 0b1'000'000'000);

  test(SV("-1"), SV("{:LB}"), -0b1);
  test(SV("-1,000,000,000"), SV("{:LB}"), -0b1'000'000'000);

  test(SV("-0B1"), SV("{:#LB}"), -0b1);
  test(SV("-0B1,000,000,000"), SV("{:#LB}"), -0b1'000'000'000);

  std::locale::global(loc);
  test(SV("0"), SV("{:Lb}"), 0b0);
  test(SV("1"), SV("{:Lb}"), 0b1);
  test(SV("1_0_00_000_00_0"), SV("{:Lb}"), 0b1'000'000'000);

  test(SV("0b0"), SV("{:#Lb}"), 0b0);
  test(SV("0b1"), SV("{:#Lb}"), 0b1);
  test(SV("0b1_0_00_000_00_0"), SV("{:#Lb}"), 0b1'000'000'000);

  test(SV("-1"), SV("{:LB}"), -0b1);
  test(SV("-1_0_00_000_00_0"), SV("{:LB}"), -0b1'000'000'000);

  test(SV("-0B1"), SV("{:#LB}"), -0b1);
  test(SV("-0B1_0_00_000_00_0"), SV("{:#LB}"), -0b1'000'000'000);

  test(SV("0"), en_US, SV("{:Lb}"), 0b0);
  test(SV("1"), en_US, SV("{:Lb}"), 0b1);
  test(SV("1,000,000,000"), en_US, SV("{:Lb}"), 0b1'000'000'000);

  test(SV("0b0"), en_US, SV("{:#Lb}"), 0b0);
  test(SV("0b1"), en_US, SV("{:#Lb}"), 0b1);
  test(SV("0b1,000,000,000"), en_US, SV("{:#Lb}"), 0b1'000'000'000);

  test(SV("-1"), en_US, SV("{:LB}"), -0b1);
  test(SV("-1,000,000,000"), en_US, SV("{:LB}"), -0b1'000'000'000);

  test(SV("-0B1"), en_US, SV("{:#LB}"), -0b1);
  test(SV("-0B1,000,000,000"), en_US, SV("{:#LB}"), -0b1'000'000'000);

  std::locale::global(en_US);
  test(SV("0"), loc, SV("{:Lb}"), 0b0);
  test(SV("1"), loc, SV("{:Lb}"), 0b1);
  test(SV("1_0_00_000_00_0"), loc, SV("{:Lb}"), 0b1'000'000'000);

  test(SV("0b0"), loc, SV("{:#Lb}"), 0b0);
  test(SV("0b1"), loc, SV("{:#Lb}"), 0b1);
  test(SV("0b1_0_00_000_00_0"), loc, SV("{:#Lb}"), 0b1'000'000'000);

  test(SV("-1"), loc, SV("{:LB}"), -0b1);
  test(SV("-1_0_00_000_00_0"), loc, SV("{:LB}"), -0b1'000'000'000);

  test(SV("-0B1"), loc, SV("{:#LB}"), -0b1);
  test(SV("-0B1_0_00_000_00_0"), loc, SV("{:#LB}"), -0b1'000'000'000);

  // *** Octal ***
  std::locale::global(en_US);
  test(SV("0"), SV("{:Lo}"), 00);
  test(SV("1"), SV("{:Lo}"), 01);
  test(SV("1,000,000,000"), SV("{:Lo}"), 01'000'000'000);

  test(SV("0"), SV("{:#Lo}"), 00);
  test(SV("01"), SV("{:#Lo}"), 01);
  test(SV("01,000,000,000"), SV("{:#Lo}"), 01'000'000'000);

  test(SV("-1"), SV("{:Lo}"), -01);
  test(SV("-1,000,000,000"), SV("{:Lo}"), -01'000'000'000);

  test(SV("-01"), SV("{:#Lo}"), -01);
  test(SV("-01,000,000,000"), SV("{:#Lo}"), -01'000'000'000);

  std::locale::global(loc);
  test(SV("0"), SV("{:Lo}"), 00);
  test(SV("1"), SV("{:Lo}"), 01);
  test(SV("1_0_00_000_00_0"), SV("{:Lo}"), 01'000'000'000);

  test(SV("0"), SV("{:#Lo}"), 00);
  test(SV("01"), SV("{:#Lo}"), 01);
  test(SV("01_0_00_000_00_0"), SV("{:#Lo}"), 01'000'000'000);

  test(SV("-1"), SV("{:Lo}"), -01);
  test(SV("-1_0_00_000_00_0"), SV("{:Lo}"), -01'000'000'000);

  test(SV("-01"), SV("{:#Lo}"), -01);
  test(SV("-01_0_00_000_00_0"), SV("{:#Lo}"), -01'000'000'000);

  test(SV("0"), en_US, SV("{:Lo}"), 00);
  test(SV("1"), en_US, SV("{:Lo}"), 01);
  test(SV("1,000,000,000"), en_US, SV("{:Lo}"), 01'000'000'000);

  test(SV("0"), en_US, SV("{:#Lo}"), 00);
  test(SV("01"), en_US, SV("{:#Lo}"), 01);
  test(SV("01,000,000,000"), en_US, SV("{:#Lo}"), 01'000'000'000);

  test(SV("-1"), en_US, SV("{:Lo}"), -01);
  test(SV("-1,000,000,000"), en_US, SV("{:Lo}"), -01'000'000'000);

  test(SV("-01"), en_US, SV("{:#Lo}"), -01);
  test(SV("-01,000,000,000"), en_US, SV("{:#Lo}"), -01'000'000'000);

  std::locale::global(en_US);
  test(SV("0"), loc, SV("{:Lo}"), 00);
  test(SV("1"), loc, SV("{:Lo}"), 01);
  test(SV("1_0_00_000_00_0"), loc, SV("{:Lo}"), 01'000'000'000);

  test(SV("0"), loc, SV("{:#Lo}"), 00);
  test(SV("01"), loc, SV("{:#Lo}"), 01);
  test(SV("01_0_00_000_00_0"), loc, SV("{:#Lo}"), 01'000'000'000);

  test(SV("-1"), loc, SV("{:Lo}"), -01);
  test(SV("-1_0_00_000_00_0"), loc, SV("{:Lo}"), -01'000'000'000);

  test(SV("-01"), loc, SV("{:#Lo}"), -01);
  test(SV("-01_0_00_000_00_0"), loc, SV("{:#Lo}"), -01'000'000'000);

  // *** Hexadecimal ***
  std::locale::global(en_US);
  test(SV("0"), SV("{:Lx}"), 0x0);
  test(SV("1"), SV("{:Lx}"), 0x1);
  test(SV("1,000,000,000"), SV("{:Lx}"), 0x1'000'000'000);

  test(SV("0x0"), SV("{:#Lx}"), 0x0);
  test(SV("0x1"), SV("{:#Lx}"), 0x1);
  test(SV("0x1,000,000,000"), SV("{:#Lx}"), 0x1'000'000'000);

  test(SV("-1"), SV("{:LX}"), -0x1);
  test(SV("-1,000,000,000"), SV("{:LX}"), -0x1'000'000'000);

  test(SV("-0X1"), SV("{:#LX}"), -0x1);
  test(SV("-0X1,000,000,000"), SV("{:#LX}"), -0x1'000'000'000);

  std::locale::global(loc);
  test(SV("0"), SV("{:Lx}"), 0x0);
  test(SV("1"), SV("{:Lx}"), 0x1);
  test(SV("1_0_00_000_00_0"), SV("{:Lx}"), 0x1'000'000'000);

  test(SV("0x0"), SV("{:#Lx}"), 0x0);
  test(SV("0x1"), SV("{:#Lx}"), 0x1);
  test(SV("0x1_0_00_000_00_0"), SV("{:#Lx}"), 0x1'000'000'000);

  test(SV("-1"), SV("{:LX}"), -0x1);
  test(SV("-1_0_00_000_00_0"), SV("{:LX}"), -0x1'000'000'000);

  test(SV("-0X1"), SV("{:#LX}"), -0x1);
  test(SV("-0X1_0_00_000_00_0"), SV("{:#LX}"), -0x1'000'000'000);

  test(SV("0"), en_US, SV("{:Lx}"), 0x0);
  test(SV("1"), en_US, SV("{:Lx}"), 0x1);
  test(SV("1,000,000,000"), en_US, SV("{:Lx}"), 0x1'000'000'000);

  test(SV("0x0"), en_US, SV("{:#Lx}"), 0x0);
  test(SV("0x1"), en_US, SV("{:#Lx}"), 0x1);
  test(SV("0x1,000,000,000"), en_US, SV("{:#Lx}"), 0x1'000'000'000);

  test(SV("-1"), en_US, SV("{:LX}"), -0x1);
  test(SV("-1,000,000,000"), en_US, SV("{:LX}"), -0x1'000'000'000);

  test(SV("-0X1"), en_US, SV("{:#LX}"), -0x1);
  test(SV("-0X1,000,000,000"), en_US, SV("{:#LX}"), -0x1'000'000'000);

  std::locale::global(en_US);
  test(SV("0"), loc, SV("{:Lx}"), 0x0);
  test(SV("1"), loc, SV("{:Lx}"), 0x1);
  test(SV("1_0_00_000_00_0"), loc, SV("{:Lx}"), 0x1'000'000'000);

  test(SV("0x0"), loc, SV("{:#Lx}"), 0x0);
  test(SV("0x1"), loc, SV("{:#Lx}"), 0x1);
  test(SV("0x1_0_00_000_00_0"), loc, SV("{:#Lx}"), 0x1'000'000'000);

  test(SV("-1"), loc, SV("{:LX}"), -0x1);
  test(SV("-1_0_00_000_00_0"), loc, SV("{:LX}"), -0x1'000'000'000);

  test(SV("-0X1"), loc, SV("{:#LX}"), -0x1);
  test(SV("-0X1_0_00_000_00_0"), loc, SV("{:#LX}"), -0x1'000'000'000);

  // *** align-fill & width ***
  test(SV("4_2"), loc, SV("{:L}"), 42);

  test(SV("   4_2"), loc, SV("{:6L}"), 42);
  test(SV("4_2   "), loc, SV("{:<6L}"), 42);
  test(SV(" 4_2  "), loc, SV("{:^6L}"), 42);
  test(SV("   4_2"), loc, SV("{:>6L}"), 42);

  test(SV("4_2***"), loc, SV("{:*<6L}"), 42);
  test(SV("*4_2**"), loc, SV("{:*^6L}"), 42);
  test(SV("***4_2"), loc, SV("{:*>6L}"), 42);

  test(SV("4_a*****"), loc, SV("{:*<8Lx}"), 0x4a);
  test(SV("**4_a***"), loc, SV("{:*^8Lx}"), 0x4a);
  test(SV("*****4_a"), loc, SV("{:*>8Lx}"), 0x4a);

  test(SV("0x4_a***"), loc, SV("{:*<#8Lx}"), 0x4a);
  test(SV("*0x4_a**"), loc, SV("{:*^#8Lx}"), 0x4a);
  test(SV("***0x4_a"), loc, SV("{:*>#8Lx}"), 0x4a);

  test(SV("4_A*****"), loc, SV("{:*<8LX}"), 0x4a);
  test(SV("**4_A***"), loc, SV("{:*^8LX}"), 0x4a);
  test(SV("*****4_A"), loc, SV("{:*>8LX}"), 0x4a);

  test(SV("0X4_A***"), loc, SV("{:*<#8LX}"), 0x4a);
  test(SV("*0X4_A**"), loc, SV("{:*^#8LX}"), 0x4a);
  test(SV("***0X4_A"), loc, SV("{:*>#8LX}"), 0x4a);

  // Test whether zero padding is ignored
  test(SV("4_2   "), loc, SV("{:<06L}"), 42);
  test(SV(" 4_2  "), loc, SV("{:^06L}"), 42);
  test(SV("   4_2"), loc, SV("{:>06L}"), 42);

  // *** zero-padding & width ***
  test(SV("   4_2"), loc, SV("{:6L}"), 42);
  test(SV("0004_2"), loc, SV("{:06L}"), 42);
  test(SV("-004_2"), loc, SV("{:06L}"), -42);

  test(SV("000004_a"), loc, SV("{:08Lx}"), 0x4a);
  test(SV("0x0004_a"), loc, SV("{:#08Lx}"), 0x4a);
  test(SV("0X0004_A"), loc, SV("{:#08LX}"), 0x4a);

  test(SV("-00004_a"), loc, SV("{:08Lx}"), -0x4a);
  test(SV("-0x004_a"), loc, SV("{:#08Lx}"), -0x4a);
  test(SV("-0X004_A"), loc, SV("{:#08LX}"), -0x4a);
}

template <class F, class CharT>
void test_floating_point_hex_lower_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(SV("1.23456p-3"), SV("{:La}"), F(0x1.23456p-3));
  test(SV("1.23456p-2"), SV("{:La}"), F(0x1.23456p-2));
  test(SV("1.23456p-1"), SV("{:La}"), F(0x1.23456p-1));
  test(SV("1.23456p+0"), SV("{:La}"), F(0x1.23456p0));
  test(SV("1.23456p+1"), SV("{:La}"), F(0x1.23456p+1));
  test(SV("1.23456p+2"), SV("{:La}"), F(0x1.23456p+2));
  test(SV("1.23456p+3"), SV("{:La}"), F(0x1.23456p+3));
  test(SV("1.23456p+20"), SV("{:La}"), F(0x1.23456p+20));

  std::locale::global(loc);
  test(SV("1#23456p-3"), SV("{:La}"), F(0x1.23456p-3));
  test(SV("1#23456p-2"), SV("{:La}"), F(0x1.23456p-2));
  test(SV("1#23456p-1"), SV("{:La}"), F(0x1.23456p-1));
  test(SV("1#23456p+0"), SV("{:La}"), F(0x1.23456p0));
  test(SV("1#23456p+1"), SV("{:La}"), F(0x1.23456p+1));
  test(SV("1#23456p+2"), SV("{:La}"), F(0x1.23456p+2));
  test(SV("1#23456p+3"), SV("{:La}"), F(0x1.23456p+3));
  test(SV("1#23456p+20"), SV("{:La}"), F(0x1.23456p+20));

  test(SV("1.23456p-3"), en_US, SV("{:La}"), F(0x1.23456p-3));
  test(SV("1.23456p-2"), en_US, SV("{:La}"), F(0x1.23456p-2));
  test(SV("1.23456p-1"), en_US, SV("{:La}"), F(0x1.23456p-1));
  test(SV("1.23456p+0"), en_US, SV("{:La}"), F(0x1.23456p0));
  test(SV("1.23456p+1"), en_US, SV("{:La}"), F(0x1.23456p+1));
  test(SV("1.23456p+2"), en_US, SV("{:La}"), F(0x1.23456p+2));
  test(SV("1.23456p+3"), en_US, SV("{:La}"), F(0x1.23456p+3));
  test(SV("1.23456p+20"), en_US, SV("{:La}"), F(0x1.23456p+20));

  std::locale::global(en_US);
  test(SV("1#23456p-3"), loc, SV("{:La}"), F(0x1.23456p-3));
  test(SV("1#23456p-2"), loc, SV("{:La}"), F(0x1.23456p-2));
  test(SV("1#23456p-1"), loc, SV("{:La}"), F(0x1.23456p-1));
  test(SV("1#23456p+0"), loc, SV("{:La}"), F(0x1.23456p0));
  test(SV("1#23456p+1"), loc, SV("{:La}"), F(0x1.23456p+1));
  test(SV("1#23456p+2"), loc, SV("{:La}"), F(0x1.23456p+2));
  test(SV("1#23456p+3"), loc, SV("{:La}"), F(0x1.23456p+3));
  test(SV("1#23456p+20"), loc, SV("{:La}"), F(0x1.23456p+20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(SV("1.23456p+3$$$"), SV("{:$<13La}"), F(0x1.23456p3));
  test(SV("$$$1.23456p+3"), SV("{:$>13La}"), F(0x1.23456p3));
  test(SV("$1.23456p+3$$"), SV("{:$^13La}"), F(0x1.23456p3));
  test(SV("0001.23456p+3"), SV("{:013La}"), F(0x1.23456p3));
  test(SV("-1.23456p+3$$$"), SV("{:$<14La}"), F(-0x1.23456p3));
  test(SV("$$$-1.23456p+3"), SV("{:$>14La}"), F(-0x1.23456p3));
  test(SV("$-1.23456p+3$$"), SV("{:$^14La}"), F(-0x1.23456p3));
  test(SV("-0001.23456p+3"), SV("{:014La}"), F(-0x1.23456p3));

  std::locale::global(loc);
  test(SV("1#23456p+3$$$"), SV("{:$<13La}"), F(0x1.23456p3));
  test(SV("$$$1#23456p+3"), SV("{:$>13La}"), F(0x1.23456p3));
  test(SV("$1#23456p+3$$"), SV("{:$^13La}"), F(0x1.23456p3));
  test(SV("0001#23456p+3"), SV("{:013La}"), F(0x1.23456p3));
  test(SV("-1#23456p+3$$$"), SV("{:$<14La}"), F(-0x1.23456p3));
  test(SV("$$$-1#23456p+3"), SV("{:$>14La}"), F(-0x1.23456p3));
  test(SV("$-1#23456p+3$$"), SV("{:$^14La}"), F(-0x1.23456p3));
  test(SV("-0001#23456p+3"), SV("{:014La}"), F(-0x1.23456p3));

  test(SV("1.23456p+3$$$"), en_US, SV("{:$<13La}"), F(0x1.23456p3));
  test(SV("$$$1.23456p+3"), en_US, SV("{:$>13La}"), F(0x1.23456p3));
  test(SV("$1.23456p+3$$"), en_US, SV("{:$^13La}"), F(0x1.23456p3));
  test(SV("0001.23456p+3"), en_US, SV("{:013La}"), F(0x1.23456p3));
  test(SV("-1.23456p+3$$$"), en_US, SV("{:$<14La}"), F(-0x1.23456p3));
  test(SV("$$$-1.23456p+3"), en_US, SV("{:$>14La}"), F(-0x1.23456p3));
  test(SV("$-1.23456p+3$$"), en_US, SV("{:$^14La}"), F(-0x1.23456p3));
  test(SV("-0001.23456p+3"), en_US, SV("{:014La}"), F(-0x1.23456p3));

  std::locale::global(en_US);
  test(SV("1#23456p+3$$$"), loc, SV("{:$<13La}"), F(0x1.23456p3));
  test(SV("$$$1#23456p+3"), loc, SV("{:$>13La}"), F(0x1.23456p3));
  test(SV("$1#23456p+3$$"), loc, SV("{:$^13La}"), F(0x1.23456p3));
  test(SV("0001#23456p+3"), loc, SV("{:013La}"), F(0x1.23456p3));
  test(SV("-1#23456p+3$$$"), loc, SV("{:$<14La}"), F(-0x1.23456p3));
  test(SV("$$$-1#23456p+3"), loc, SV("{:$>14La}"), F(-0x1.23456p3));
  test(SV("$-1#23456p+3$$"), loc, SV("{:$^14La}"), F(-0x1.23456p3));
  test(SV("-0001#23456p+3"), loc, SV("{:014La}"), F(-0x1.23456p3));
}

template <class F, class CharT>
void test_floating_point_hex_upper_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(SV("1.23456P-3"), SV("{:LA}"), F(0x1.23456p-3));
  test(SV("1.23456P-2"), SV("{:LA}"), F(0x1.23456p-2));
  test(SV("1.23456P-1"), SV("{:LA}"), F(0x1.23456p-1));
  test(SV("1.23456P+0"), SV("{:LA}"), F(0x1.23456p0));
  test(SV("1.23456P+1"), SV("{:LA}"), F(0x1.23456p+1));
  test(SV("1.23456P+2"), SV("{:LA}"), F(0x1.23456p+2));
  test(SV("1.23456P+3"), SV("{:LA}"), F(0x1.23456p+3));
  test(SV("1.23456P+20"), SV("{:LA}"), F(0x1.23456p+20));

  std::locale::global(loc);
  test(SV("1#23456P-3"), SV("{:LA}"), F(0x1.23456p-3));
  test(SV("1#23456P-2"), SV("{:LA}"), F(0x1.23456p-2));
  test(SV("1#23456P-1"), SV("{:LA}"), F(0x1.23456p-1));
  test(SV("1#23456P+0"), SV("{:LA}"), F(0x1.23456p0));
  test(SV("1#23456P+1"), SV("{:LA}"), F(0x1.23456p+1));
  test(SV("1#23456P+2"), SV("{:LA}"), F(0x1.23456p+2));
  test(SV("1#23456P+3"), SV("{:LA}"), F(0x1.23456p+3));
  test(SV("1#23456P+20"), SV("{:LA}"), F(0x1.23456p+20));

  test(SV("1.23456P-3"), en_US, SV("{:LA}"), F(0x1.23456p-3));
  test(SV("1.23456P-2"), en_US, SV("{:LA}"), F(0x1.23456p-2));
  test(SV("1.23456P-1"), en_US, SV("{:LA}"), F(0x1.23456p-1));
  test(SV("1.23456P+0"), en_US, SV("{:LA}"), F(0x1.23456p0));
  test(SV("1.23456P+1"), en_US, SV("{:LA}"), F(0x1.23456p+1));
  test(SV("1.23456P+2"), en_US, SV("{:LA}"), F(0x1.23456p+2));
  test(SV("1.23456P+3"), en_US, SV("{:LA}"), F(0x1.23456p+3));
  test(SV("1.23456P+20"), en_US, SV("{:LA}"), F(0x1.23456p+20));

  std::locale::global(en_US);
  test(SV("1#23456P-3"), loc, SV("{:LA}"), F(0x1.23456p-3));
  test(SV("1#23456P-2"), loc, SV("{:LA}"), F(0x1.23456p-2));
  test(SV("1#23456P-1"), loc, SV("{:LA}"), F(0x1.23456p-1));
  test(SV("1#23456P+0"), loc, SV("{:LA}"), F(0x1.23456p0));
  test(SV("1#23456P+1"), loc, SV("{:LA}"), F(0x1.23456p+1));
  test(SV("1#23456P+2"), loc, SV("{:LA}"), F(0x1.23456p+2));
  test(SV("1#23456P+3"), loc, SV("{:LA}"), F(0x1.23456p+3));
  test(SV("1#23456P+20"), loc, SV("{:LA}"), F(0x1.23456p+20));

  // *** Fill, align, zero Padding ***
  std::locale::global(en_US);
  test(SV("1.23456P+3$$$"), SV("{:$<13LA}"), F(0x1.23456p3));
  test(SV("$$$1.23456P+3"), SV("{:$>13LA}"), F(0x1.23456p3));
  test(SV("$1.23456P+3$$"), SV("{:$^13LA}"), F(0x1.23456p3));
  test(SV("0001.23456P+3"), SV("{:013LA}"), F(0x1.23456p3));
  test(SV("-1.23456P+3$$$"), SV("{:$<14LA}"), F(-0x1.23456p3));
  test(SV("$$$-1.23456P+3"), SV("{:$>14LA}"), F(-0x1.23456p3));
  test(SV("$-1.23456P+3$$"), SV("{:$^14LA}"), F(-0x1.23456p3));
  test(SV("-0001.23456P+3"), SV("{:014LA}"), F(-0x1.23456p3));

  std::locale::global(loc);
  test(SV("1#23456P+3$$$"), SV("{:$<13LA}"), F(0x1.23456p3));
  test(SV("$$$1#23456P+3"), SV("{:$>13LA}"), F(0x1.23456p3));
  test(SV("$1#23456P+3$$"), SV("{:$^13LA}"), F(0x1.23456p3));
  test(SV("0001#23456P+3"), SV("{:013LA}"), F(0x1.23456p3));
  test(SV("-1#23456P+3$$$"), SV("{:$<14LA}"), F(-0x1.23456p3));
  test(SV("$$$-1#23456P+3"), SV("{:$>14LA}"), F(-0x1.23456p3));
  test(SV("$-1#23456P+3$$"), SV("{:$^14LA}"), F(-0x1.23456p3));
  test(SV("-0001#23456P+3"), SV("{:014LA}"), F(-0x1.23456p3));

  test(SV("1.23456P+3$$$"), en_US, SV("{:$<13LA}"), F(0x1.23456p3));
  test(SV("$$$1.23456P+3"), en_US, SV("{:$>13LA}"), F(0x1.23456p3));
  test(SV("$1.23456P+3$$"), en_US, SV("{:$^13LA}"), F(0x1.23456p3));
  test(SV("0001.23456P+3"), en_US, SV("{:013LA}"), F(0x1.23456p3));
  test(SV("-1.23456P+3$$$"), en_US, SV("{:$<14LA}"), F(-0x1.23456p3));
  test(SV("$$$-1.23456P+3"), en_US, SV("{:$>14LA}"), F(-0x1.23456p3));
  test(SV("$-1.23456P+3$$"), en_US, SV("{:$^14LA}"), F(-0x1.23456p3));
  test(SV("-0001.23456P+3"), en_US, SV("{:014LA}"), F(-0x1.23456p3));

  std::locale::global(en_US);
  test(SV("1#23456P+3$$$"), loc, SV("{:$<13LA}"), F(0x1.23456p3));
  test(SV("$$$1#23456P+3"), loc, SV("{:$>13LA}"), F(0x1.23456p3));
  test(SV("$1#23456P+3$$"), loc, SV("{:$^13LA}"), F(0x1.23456p3));
  test(SV("0001#23456P+3"), loc, SV("{:013LA}"), F(0x1.23456p3));
  test(SV("-1#23456P+3$$$"), loc, SV("{:$<14LA}"), F(-0x1.23456p3));
  test(SV("$$$-1#23456P+3"), loc, SV("{:$>14LA}"), F(-0x1.23456p3));
  test(SV("$-1#23456P+3$$"), loc, SV("{:$^14LA}"), F(-0x1.23456p3));
  test(SV("-0001#23456P+3"), loc, SV("{:014LA}"), F(-0x1.23456p3));
}

template <class F, class CharT>
void test_floating_point_hex_lower_case_precision() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(SV("1.234560p-3"), SV("{:.6La}"), F(0x1.23456p-3));
  test(SV("1.234560p-2"), SV("{:.6La}"), F(0x1.23456p-2));
  test(SV("1.234560p-1"), SV("{:.6La}"), F(0x1.23456p-1));
  test(SV("1.234560p+0"), SV("{:.6La}"), F(0x1.23456p0));
  test(SV("1.234560p+1"), SV("{:.6La}"), F(0x1.23456p+1));
  test(SV("1.234560p+2"), SV("{:.6La}"), F(0x1.23456p+2));
  test(SV("1.234560p+3"), SV("{:.6La}"), F(0x1.23456p+3));
  test(SV("1.234560p+20"), SV("{:.6La}"), F(0x1.23456p+20));

  std::locale::global(loc);
  test(SV("1#234560p-3"), SV("{:.6La}"), F(0x1.23456p-3));
  test(SV("1#234560p-2"), SV("{:.6La}"), F(0x1.23456p-2));
  test(SV("1#234560p-1"), SV("{:.6La}"), F(0x1.23456p-1));
  test(SV("1#234560p+0"), SV("{:.6La}"), F(0x1.23456p0));
  test(SV("1#234560p+1"), SV("{:.6La}"), F(0x1.23456p+1));
  test(SV("1#234560p+2"), SV("{:.6La}"), F(0x1.23456p+2));
  test(SV("1#234560p+3"), SV("{:.6La}"), F(0x1.23456p+3));
  test(SV("1#234560p+20"), SV("{:.6La}"), F(0x1.23456p+20));

  test(SV("1.234560p-3"), en_US, SV("{:.6La}"), F(0x1.23456p-3));
  test(SV("1.234560p-2"), en_US, SV("{:.6La}"), F(0x1.23456p-2));
  test(SV("1.234560p-1"), en_US, SV("{:.6La}"), F(0x1.23456p-1));
  test(SV("1.234560p+0"), en_US, SV("{:.6La}"), F(0x1.23456p0));
  test(SV("1.234560p+1"), en_US, SV("{:.6La}"), F(0x1.23456p+1));
  test(SV("1.234560p+2"), en_US, SV("{:.6La}"), F(0x1.23456p+2));
  test(SV("1.234560p+3"), en_US, SV("{:.6La}"), F(0x1.23456p+3));
  test(SV("1.234560p+20"), en_US, SV("{:.6La}"), F(0x1.23456p+20));

  std::locale::global(en_US);
  test(SV("1#234560p-3"), loc, SV("{:.6La}"), F(0x1.23456p-3));
  test(SV("1#234560p-2"), loc, SV("{:.6La}"), F(0x1.23456p-2));
  test(SV("1#234560p-1"), loc, SV("{:.6La}"), F(0x1.23456p-1));
  test(SV("1#234560p+0"), loc, SV("{:.6La}"), F(0x1.23456p0));
  test(SV("1#234560p+1"), loc, SV("{:.6La}"), F(0x1.23456p+1));
  test(SV("1#234560p+2"), loc, SV("{:.6La}"), F(0x1.23456p+2));
  test(SV("1#234560p+3"), loc, SV("{:.6La}"), F(0x1.23456p+3));
  test(SV("1#234560p+20"), loc, SV("{:.6La}"), F(0x1.23456p+20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(SV("1.234560p+3$$$"), SV("{:$<14.6La}"), F(0x1.23456p3));
  test(SV("$$$1.234560p+3"), SV("{:$>14.6La}"), F(0x1.23456p3));
  test(SV("$1.234560p+3$$"), SV("{:$^14.6La}"), F(0x1.23456p3));
  test(SV("0001.234560p+3"), SV("{:014.6La}"), F(0x1.23456p3));
  test(SV("-1.234560p+3$$$"), SV("{:$<15.6La}"), F(-0x1.23456p3));
  test(SV("$$$-1.234560p+3"), SV("{:$>15.6La}"), F(-0x1.23456p3));
  test(SV("$-1.234560p+3$$"), SV("{:$^15.6La}"), F(-0x1.23456p3));
  test(SV("-0001.234560p+3"), SV("{:015.6La}"), F(-0x1.23456p3));

  std::locale::global(loc);
  test(SV("1#234560p+3$$$"), SV("{:$<14.6La}"), F(0x1.23456p3));
  test(SV("$$$1#234560p+3"), SV("{:$>14.6La}"), F(0x1.23456p3));
  test(SV("$1#234560p+3$$"), SV("{:$^14.6La}"), F(0x1.23456p3));
  test(SV("0001#234560p+3"), SV("{:014.6La}"), F(0x1.23456p3));
  test(SV("-1#234560p+3$$$"), SV("{:$<15.6La}"), F(-0x1.23456p3));
  test(SV("$$$-1#234560p+3"), SV("{:$>15.6La}"), F(-0x1.23456p3));
  test(SV("$-1#234560p+3$$"), SV("{:$^15.6La}"), F(-0x1.23456p3));
  test(SV("-0001#234560p+3"), SV("{:015.6La}"), F(-0x1.23456p3));

  test(SV("1.234560p+3$$$"), en_US, SV("{:$<14.6La}"), F(0x1.23456p3));
  test(SV("$$$1.234560p+3"), en_US, SV("{:$>14.6La}"), F(0x1.23456p3));
  test(SV("$1.234560p+3$$"), en_US, SV("{:$^14.6La}"), F(0x1.23456p3));
  test(SV("0001.234560p+3"), en_US, SV("{:014.6La}"), F(0x1.23456p3));
  test(SV("-1.234560p+3$$$"), en_US, SV("{:$<15.6La}"), F(-0x1.23456p3));
  test(SV("$$$-1.234560p+3"), en_US, SV("{:$>15.6La}"), F(-0x1.23456p3));
  test(SV("$-1.234560p+3$$"), en_US, SV("{:$^15.6La}"), F(-0x1.23456p3));
  test(SV("-0001.234560p+3"), en_US, SV("{:015.6La}"), F(-0x1.23456p3));

  std::locale::global(en_US);
  test(SV("1#234560p+3$$$"), loc, SV("{:$<14.6La}"), F(0x1.23456p3));
  test(SV("$$$1#234560p+3"), loc, SV("{:$>14.6La}"), F(0x1.23456p3));
  test(SV("$1#234560p+3$$"), loc, SV("{:$^14.6La}"), F(0x1.23456p3));
  test(SV("0001#234560p+3"), loc, SV("{:014.6La}"), F(0x1.23456p3));
  test(SV("-1#234560p+3$$$"), loc, SV("{:$<15.6La}"), F(-0x1.23456p3));
  test(SV("$$$-1#234560p+3"), loc, SV("{:$>15.6La}"), F(-0x1.23456p3));
  test(SV("$-1#234560p+3$$"), loc, SV("{:$^15.6La}"), F(-0x1.23456p3));
  test(SV("-0001#234560p+3"), loc, SV("{:015.6La}"), F(-0x1.23456p3));
}

template <class F, class CharT>
void test_floating_point_hex_upper_case_precision() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(SV("1.234560P-3"), SV("{:.6LA}"), F(0x1.23456p-3));
  test(SV("1.234560P-2"), SV("{:.6LA}"), F(0x1.23456p-2));
  test(SV("1.234560P-1"), SV("{:.6LA}"), F(0x1.23456p-1));
  test(SV("1.234560P+0"), SV("{:.6LA}"), F(0x1.23456p0));
  test(SV("1.234560P+1"), SV("{:.6LA}"), F(0x1.23456p+1));
  test(SV("1.234560P+2"), SV("{:.6LA}"), F(0x1.23456p+2));
  test(SV("1.234560P+3"), SV("{:.6LA}"), F(0x1.23456p+3));
  test(SV("1.234560P+20"), SV("{:.6LA}"), F(0x1.23456p+20));

  std::locale::global(loc);
  test(SV("1#234560P-3"), SV("{:.6LA}"), F(0x1.23456p-3));
  test(SV("1#234560P-2"), SV("{:.6LA}"), F(0x1.23456p-2));
  test(SV("1#234560P-1"), SV("{:.6LA}"), F(0x1.23456p-1));
  test(SV("1#234560P+0"), SV("{:.6LA}"), F(0x1.23456p0));
  test(SV("1#234560P+1"), SV("{:.6LA}"), F(0x1.23456p+1));
  test(SV("1#234560P+2"), SV("{:.6LA}"), F(0x1.23456p+2));
  test(SV("1#234560P+3"), SV("{:.6LA}"), F(0x1.23456p+3));
  test(SV("1#234560P+20"), SV("{:.6LA}"), F(0x1.23456p+20));

  test(SV("1.234560P-3"), en_US, SV("{:.6LA}"), F(0x1.23456p-3));
  test(SV("1.234560P-2"), en_US, SV("{:.6LA}"), F(0x1.23456p-2));
  test(SV("1.234560P-1"), en_US, SV("{:.6LA}"), F(0x1.23456p-1));
  test(SV("1.234560P+0"), en_US, SV("{:.6LA}"), F(0x1.23456p0));
  test(SV("1.234560P+1"), en_US, SV("{:.6LA}"), F(0x1.23456p+1));
  test(SV("1.234560P+2"), en_US, SV("{:.6LA}"), F(0x1.23456p+2));
  test(SV("1.234560P+3"), en_US, SV("{:.6LA}"), F(0x1.23456p+3));
  test(SV("1.234560P+20"), en_US, SV("{:.6LA}"), F(0x1.23456p+20));

  std::locale::global(en_US);
  test(SV("1#234560P-3"), loc, SV("{:.6LA}"), F(0x1.23456p-3));
  test(SV("1#234560P-2"), loc, SV("{:.6LA}"), F(0x1.23456p-2));
  test(SV("1#234560P-1"), loc, SV("{:.6LA}"), F(0x1.23456p-1));
  test(SV("1#234560P+0"), loc, SV("{:.6LA}"), F(0x1.23456p0));
  test(SV("1#234560P+1"), loc, SV("{:.6LA}"), F(0x1.23456p+1));
  test(SV("1#234560P+2"), loc, SV("{:.6LA}"), F(0x1.23456p+2));
  test(SV("1#234560P+3"), loc, SV("{:.6LA}"), F(0x1.23456p+3));
  test(SV("1#234560P+20"), loc, SV("{:.6LA}"), F(0x1.23456p+20));

  // *** Fill, align, zero Padding ***
  std::locale::global(en_US);
  test(SV("1.234560P+3$$$"), SV("{:$<14.6LA}"), F(0x1.23456p3));
  test(SV("$$$1.234560P+3"), SV("{:$>14.6LA}"), F(0x1.23456p3));
  test(SV("$1.234560P+3$$"), SV("{:$^14.6LA}"), F(0x1.23456p3));
  test(SV("0001.234560P+3"), SV("{:014.6LA}"), F(0x1.23456p3));
  test(SV("-1.234560P+3$$$"), SV("{:$<15.6LA}"), F(-0x1.23456p3));
  test(SV("$$$-1.234560P+3"), SV("{:$>15.6LA}"), F(-0x1.23456p3));
  test(SV("$-1.234560P+3$$"), SV("{:$^15.6LA}"), F(-0x1.23456p3));
  test(SV("-0001.234560P+3"), SV("{:015.6LA}"), F(-0x1.23456p3));

  std::locale::global(loc);
  test(SV("1#234560P+3$$$"), SV("{:$<14.6LA}"), F(0x1.23456p3));
  test(SV("$$$1#234560P+3"), SV("{:$>14.6LA}"), F(0x1.23456p3));
  test(SV("$1#234560P+3$$"), SV("{:$^14.6LA}"), F(0x1.23456p3));
  test(SV("0001#234560P+3"), SV("{:014.6LA}"), F(0x1.23456p3));
  test(SV("-1#234560P+3$$$"), SV("{:$<15.6LA}"), F(-0x1.23456p3));
  test(SV("$$$-1#234560P+3"), SV("{:$>15.6LA}"), F(-0x1.23456p3));
  test(SV("$-1#234560P+3$$"), SV("{:$^15.6LA}"), F(-0x1.23456p3));
  test(SV("-0001#234560P+3"), SV("{:015.6LA}"), F(-0x1.23456p3));

  test(SV("1.234560P+3$$$"), en_US, SV("{:$<14.6LA}"), F(0x1.23456p3));
  test(SV("$$$1.234560P+3"), en_US, SV("{:$>14.6LA}"), F(0x1.23456p3));
  test(SV("$1.234560P+3$$"), en_US, SV("{:$^14.6LA}"), F(0x1.23456p3));
  test(SV("0001.234560P+3"), en_US, SV("{:014.6LA}"), F(0x1.23456p3));
  test(SV("-1.234560P+3$$$"), en_US, SV("{:$<15.6LA}"), F(-0x1.23456p3));
  test(SV("$$$-1.234560P+3"), en_US, SV("{:$>15.6LA}"), F(-0x1.23456p3));
  test(SV("$-1.234560P+3$$"), en_US, SV("{:$^15.6LA}"), F(-0x1.23456p3));
  test(SV("-0001.234560P+3"), en_US, SV("{:015.6LA}"), F(-0x1.23456p3));

  std::locale::global(en_US);
  test(SV("1#234560P+3$$$"), loc, SV("{:$<14.6LA}"), F(0x1.23456p3));
  test(SV("$$$1#234560P+3"), loc, SV("{:$>14.6LA}"), F(0x1.23456p3));
  test(SV("$1#234560P+3$$"), loc, SV("{:$^14.6LA}"), F(0x1.23456p3));
  test(SV("0001#234560P+3"), loc, SV("{:014.6LA}"), F(0x1.23456p3));
  test(SV("-1#234560P+3$$$"), loc, SV("{:$<15.6LA}"), F(-0x1.23456p3));
  test(SV("$$$-1#234560P+3"), loc, SV("{:$>15.6LA}"), F(-0x1.23456p3));
  test(SV("$-1#234560P+3$$"), loc, SV("{:$^15.6LA}"), F(-0x1.23456p3));
  test(SV("-0001#234560P+3"), loc, SV("{:015.6LA}"), F(-0x1.23456p3));
}

template <class F, class CharT>
void test_floating_point_scientific_lower_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(SV("1.234567e-03"), SV("{:.6Le}"), F(1.234567e-3));
  test(SV("1.234567e-02"), SV("{:.6Le}"), F(1.234567e-2));
  test(SV("1.234567e-01"), SV("{:.6Le}"), F(1.234567e-1));
  test(SV("1.234567e+00"), SV("{:.6Le}"), F(1.234567e0));
  test(SV("1.234567e+01"), SV("{:.6Le}"), F(1.234567e1));
  test(SV("1.234567e+02"), SV("{:.6Le}"), F(1.234567e2));
  test(SV("1.234567e+03"), SV("{:.6Le}"), F(1.234567e3));
  test(SV("1.234567e+20"), SV("{:.6Le}"), F(1.234567e20));
  test(SV("-1.234567e-03"), SV("{:.6Le}"), F(-1.234567e-3));
  test(SV("-1.234567e-02"), SV("{:.6Le}"), F(-1.234567e-2));
  test(SV("-1.234567e-01"), SV("{:.6Le}"), F(-1.234567e-1));
  test(SV("-1.234567e+00"), SV("{:.6Le}"), F(-1.234567e0));
  test(SV("-1.234567e+01"), SV("{:.6Le}"), F(-1.234567e1));
  test(SV("-1.234567e+02"), SV("{:.6Le}"), F(-1.234567e2));
  test(SV("-1.234567e+03"), SV("{:.6Le}"), F(-1.234567e3));
  test(SV("-1.234567e+20"), SV("{:.6Le}"), F(-1.234567e20));

  std::locale::global(loc);
  test(SV("1#234567e-03"), SV("{:.6Le}"), F(1.234567e-3));
  test(SV("1#234567e-02"), SV("{:.6Le}"), F(1.234567e-2));
  test(SV("1#234567e-01"), SV("{:.6Le}"), F(1.234567e-1));
  test(SV("1#234567e+00"), SV("{:.6Le}"), F(1.234567e0));
  test(SV("1#234567e+01"), SV("{:.6Le}"), F(1.234567e1));
  test(SV("1#234567e+02"), SV("{:.6Le}"), F(1.234567e2));
  test(SV("1#234567e+03"), SV("{:.6Le}"), F(1.234567e3));
  test(SV("1#234567e+20"), SV("{:.6Le}"), F(1.234567e20));
  test(SV("-1#234567e-03"), SV("{:.6Le}"), F(-1.234567e-3));
  test(SV("-1#234567e-02"), SV("{:.6Le}"), F(-1.234567e-2));
  test(SV("-1#234567e-01"), SV("{:.6Le}"), F(-1.234567e-1));
  test(SV("-1#234567e+00"), SV("{:.6Le}"), F(-1.234567e0));
  test(SV("-1#234567e+01"), SV("{:.6Le}"), F(-1.234567e1));
  test(SV("-1#234567e+02"), SV("{:.6Le}"), F(-1.234567e2));
  test(SV("-1#234567e+03"), SV("{:.6Le}"), F(-1.234567e3));
  test(SV("-1#234567e+20"), SV("{:.6Le}"), F(-1.234567e20));

  test(SV("1.234567e-03"), en_US, SV("{:.6Le}"), F(1.234567e-3));
  test(SV("1.234567e-02"), en_US, SV("{:.6Le}"), F(1.234567e-2));
  test(SV("1.234567e-01"), en_US, SV("{:.6Le}"), F(1.234567e-1));
  test(SV("1.234567e+00"), en_US, SV("{:.6Le}"), F(1.234567e0));
  test(SV("1.234567e+01"), en_US, SV("{:.6Le}"), F(1.234567e1));
  test(SV("1.234567e+02"), en_US, SV("{:.6Le}"), F(1.234567e2));
  test(SV("1.234567e+03"), en_US, SV("{:.6Le}"), F(1.234567e3));
  test(SV("1.234567e+20"), en_US, SV("{:.6Le}"), F(1.234567e20));
  test(SV("-1.234567e-03"), en_US, SV("{:.6Le}"), F(-1.234567e-3));
  test(SV("-1.234567e-02"), en_US, SV("{:.6Le}"), F(-1.234567e-2));
  test(SV("-1.234567e-01"), en_US, SV("{:.6Le}"), F(-1.234567e-1));
  test(SV("-1.234567e+00"), en_US, SV("{:.6Le}"), F(-1.234567e0));
  test(SV("-1.234567e+01"), en_US, SV("{:.6Le}"), F(-1.234567e1));
  test(SV("-1.234567e+02"), en_US, SV("{:.6Le}"), F(-1.234567e2));
  test(SV("-1.234567e+03"), en_US, SV("{:.6Le}"), F(-1.234567e3));
  test(SV("-1.234567e+20"), en_US, SV("{:.6Le}"), F(-1.234567e20));

  std::locale::global(en_US);
  test(SV("1#234567e-03"), loc, SV("{:.6Le}"), F(1.234567e-3));
  test(SV("1#234567e-02"), loc, SV("{:.6Le}"), F(1.234567e-2));
  test(SV("1#234567e-01"), loc, SV("{:.6Le}"), F(1.234567e-1));
  test(SV("1#234567e+00"), loc, SV("{:.6Le}"), F(1.234567e0));
  test(SV("1#234567e+01"), loc, SV("{:.6Le}"), F(1.234567e1));
  test(SV("1#234567e+02"), loc, SV("{:.6Le}"), F(1.234567e2));
  test(SV("1#234567e+03"), loc, SV("{:.6Le}"), F(1.234567e3));
  test(SV("1#234567e+20"), loc, SV("{:.6Le}"), F(1.234567e20));
  test(SV("-1#234567e-03"), loc, SV("{:.6Le}"), F(-1.234567e-3));
  test(SV("-1#234567e-02"), loc, SV("{:.6Le}"), F(-1.234567e-2));
  test(SV("-1#234567e-01"), loc, SV("{:.6Le}"), F(-1.234567e-1));
  test(SV("-1#234567e+00"), loc, SV("{:.6Le}"), F(-1.234567e0));
  test(SV("-1#234567e+01"), loc, SV("{:.6Le}"), F(-1.234567e1));
  test(SV("-1#234567e+02"), loc, SV("{:.6Le}"), F(-1.234567e2));
  test(SV("-1#234567e+03"), loc, SV("{:.6Le}"), F(-1.234567e3));
  test(SV("-1#234567e+20"), loc, SV("{:.6Le}"), F(-1.234567e20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(SV("1.234567e+03$$$"), SV("{:$<15.6Le}"), F(1.234567e3));
  test(SV("$$$1.234567e+03"), SV("{:$>15.6Le}"), F(1.234567e3));
  test(SV("$1.234567e+03$$"), SV("{:$^15.6Le}"), F(1.234567e3));
  test(SV("0001.234567e+03"), SV("{:015.6Le}"), F(1.234567e3));
  test(SV("-1.234567e+03$$$"), SV("{:$<16.6Le}"), F(-1.234567e3));
  test(SV("$$$-1.234567e+03"), SV("{:$>16.6Le}"), F(-1.234567e3));
  test(SV("$-1.234567e+03$$"), SV("{:$^16.6Le}"), F(-1.234567e3));
  test(SV("-0001.234567e+03"), SV("{:016.6Le}"), F(-1.234567e3));

  std::locale::global(loc);
  test(SV("1#234567e+03$$$"), SV("{:$<15.6Le}"), F(1.234567e3));
  test(SV("$$$1#234567e+03"), SV("{:$>15.6Le}"), F(1.234567e3));
  test(SV("$1#234567e+03$$"), SV("{:$^15.6Le}"), F(1.234567e3));
  test(SV("0001#234567e+03"), SV("{:015.6Le}"), F(1.234567e3));
  test(SV("-1#234567e+03$$$"), SV("{:$<16.6Le}"), F(-1.234567e3));
  test(SV("$$$-1#234567e+03"), SV("{:$>16.6Le}"), F(-1.234567e3));
  test(SV("$-1#234567e+03$$"), SV("{:$^16.6Le}"), F(-1.234567e3));
  test(SV("-0001#234567e+03"), SV("{:016.6Le}"), F(-1.234567e3));

  test(SV("1.234567e+03$$$"), en_US, SV("{:$<15.6Le}"), F(1.234567e3));
  test(SV("$$$1.234567e+03"), en_US, SV("{:$>15.6Le}"), F(1.234567e3));
  test(SV("$1.234567e+03$$"), en_US, SV("{:$^15.6Le}"), F(1.234567e3));
  test(SV("0001.234567e+03"), en_US, SV("{:015.6Le}"), F(1.234567e3));
  test(SV("-1.234567e+03$$$"), en_US, SV("{:$<16.6Le}"), F(-1.234567e3));
  test(SV("$$$-1.234567e+03"), en_US, SV("{:$>16.6Le}"), F(-1.234567e3));
  test(SV("$-1.234567e+03$$"), en_US, SV("{:$^16.6Le}"), F(-1.234567e3));
  test(SV("-0001.234567e+03"), en_US, SV("{:016.6Le}"), F(-1.234567e3));

  std::locale::global(en_US);
  test(SV("1#234567e+03$$$"), loc, SV("{:$<15.6Le}"), F(1.234567e3));
  test(SV("$$$1#234567e+03"), loc, SV("{:$>15.6Le}"), F(1.234567e3));
  test(SV("$1#234567e+03$$"), loc, SV("{:$^15.6Le}"), F(1.234567e3));
  test(SV("0001#234567e+03"), loc, SV("{:015.6Le}"), F(1.234567e3));
  test(SV("-1#234567e+03$$$"), loc, SV("{:$<16.6Le}"), F(-1.234567e3));
  test(SV("$$$-1#234567e+03"), loc, SV("{:$>16.6Le}"), F(-1.234567e3));
  test(SV("$-1#234567e+03$$"), loc, SV("{:$^16.6Le}"), F(-1.234567e3));
  test(SV("-0001#234567e+03"), loc, SV("{:016.6Le}"), F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_scientific_upper_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(SV("1.234567E-03"), SV("{:.6LE}"), F(1.234567e-3));
  test(SV("1.234567E-02"), SV("{:.6LE}"), F(1.234567e-2));
  test(SV("1.234567E-01"), SV("{:.6LE}"), F(1.234567e-1));
  test(SV("1.234567E+00"), SV("{:.6LE}"), F(1.234567e0));
  test(SV("1.234567E+01"), SV("{:.6LE}"), F(1.234567e1));
  test(SV("1.234567E+02"), SV("{:.6LE}"), F(1.234567e2));
  test(SV("1.234567E+03"), SV("{:.6LE}"), F(1.234567e3));
  test(SV("1.234567E+20"), SV("{:.6LE}"), F(1.234567e20));
  test(SV("-1.234567E-03"), SV("{:.6LE}"), F(-1.234567e-3));
  test(SV("-1.234567E-02"), SV("{:.6LE}"), F(-1.234567e-2));
  test(SV("-1.234567E-01"), SV("{:.6LE}"), F(-1.234567e-1));
  test(SV("-1.234567E+00"), SV("{:.6LE}"), F(-1.234567e0));
  test(SV("-1.234567E+01"), SV("{:.6LE}"), F(-1.234567e1));
  test(SV("-1.234567E+02"), SV("{:.6LE}"), F(-1.234567e2));
  test(SV("-1.234567E+03"), SV("{:.6LE}"), F(-1.234567e3));
  test(SV("-1.234567E+20"), SV("{:.6LE}"), F(-1.234567e20));

  std::locale::global(loc);
  test(SV("1#234567E-03"), SV("{:.6LE}"), F(1.234567e-3));
  test(SV("1#234567E-02"), SV("{:.6LE}"), F(1.234567e-2));
  test(SV("1#234567E-01"), SV("{:.6LE}"), F(1.234567e-1));
  test(SV("1#234567E+00"), SV("{:.6LE}"), F(1.234567e0));
  test(SV("1#234567E+01"), SV("{:.6LE}"), F(1.234567e1));
  test(SV("1#234567E+02"), SV("{:.6LE}"), F(1.234567e2));
  test(SV("1#234567E+03"), SV("{:.6LE}"), F(1.234567e3));
  test(SV("1#234567E+20"), SV("{:.6LE}"), F(1.234567e20));
  test(SV("-1#234567E-03"), SV("{:.6LE}"), F(-1.234567e-3));
  test(SV("-1#234567E-02"), SV("{:.6LE}"), F(-1.234567e-2));
  test(SV("-1#234567E-01"), SV("{:.6LE}"), F(-1.234567e-1));
  test(SV("-1#234567E+00"), SV("{:.6LE}"), F(-1.234567e0));
  test(SV("-1#234567E+01"), SV("{:.6LE}"), F(-1.234567e1));
  test(SV("-1#234567E+02"), SV("{:.6LE}"), F(-1.234567e2));
  test(SV("-1#234567E+03"), SV("{:.6LE}"), F(-1.234567e3));
  test(SV("-1#234567E+20"), SV("{:.6LE}"), F(-1.234567e20));

  test(SV("1.234567E-03"), en_US, SV("{:.6LE}"), F(1.234567e-3));
  test(SV("1.234567E-02"), en_US, SV("{:.6LE}"), F(1.234567e-2));
  test(SV("1.234567E-01"), en_US, SV("{:.6LE}"), F(1.234567e-1));
  test(SV("1.234567E+00"), en_US, SV("{:.6LE}"), F(1.234567e0));
  test(SV("1.234567E+01"), en_US, SV("{:.6LE}"), F(1.234567e1));
  test(SV("1.234567E+02"), en_US, SV("{:.6LE}"), F(1.234567e2));
  test(SV("1.234567E+03"), en_US, SV("{:.6LE}"), F(1.234567e3));
  test(SV("1.234567E+20"), en_US, SV("{:.6LE}"), F(1.234567e20));
  test(SV("-1.234567E-03"), en_US, SV("{:.6LE}"), F(-1.234567e-3));
  test(SV("-1.234567E-02"), en_US, SV("{:.6LE}"), F(-1.234567e-2));
  test(SV("-1.234567E-01"), en_US, SV("{:.6LE}"), F(-1.234567e-1));
  test(SV("-1.234567E+00"), en_US, SV("{:.6LE}"), F(-1.234567e0));
  test(SV("-1.234567E+01"), en_US, SV("{:.6LE}"), F(-1.234567e1));
  test(SV("-1.234567E+02"), en_US, SV("{:.6LE}"), F(-1.234567e2));
  test(SV("-1.234567E+03"), en_US, SV("{:.6LE}"), F(-1.234567e3));
  test(SV("-1.234567E+20"), en_US, SV("{:.6LE}"), F(-1.234567e20));

  std::locale::global(en_US);
  test(SV("1#234567E-03"), loc, SV("{:.6LE}"), F(1.234567e-3));
  test(SV("1#234567E-02"), loc, SV("{:.6LE}"), F(1.234567e-2));
  test(SV("1#234567E-01"), loc, SV("{:.6LE}"), F(1.234567e-1));
  test(SV("1#234567E+00"), loc, SV("{:.6LE}"), F(1.234567e0));
  test(SV("1#234567E+01"), loc, SV("{:.6LE}"), F(1.234567e1));
  test(SV("1#234567E+02"), loc, SV("{:.6LE}"), F(1.234567e2));
  test(SV("1#234567E+03"), loc, SV("{:.6LE}"), F(1.234567e3));
  test(SV("1#234567E+20"), loc, SV("{:.6LE}"), F(1.234567e20));
  test(SV("-1#234567E-03"), loc, SV("{:.6LE}"), F(-1.234567e-3));
  test(SV("-1#234567E-02"), loc, SV("{:.6LE}"), F(-1.234567e-2));
  test(SV("-1#234567E-01"), loc, SV("{:.6LE}"), F(-1.234567e-1));
  test(SV("-1#234567E+00"), loc, SV("{:.6LE}"), F(-1.234567e0));
  test(SV("-1#234567E+01"), loc, SV("{:.6LE}"), F(-1.234567e1));
  test(SV("-1#234567E+02"), loc, SV("{:.6LE}"), F(-1.234567e2));
  test(SV("-1#234567E+03"), loc, SV("{:.6LE}"), F(-1.234567e3));
  test(SV("-1#234567E+20"), loc, SV("{:.6LE}"), F(-1.234567e20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(SV("1.234567E+03$$$"), SV("{:$<15.6LE}"), F(1.234567e3));
  test(SV("$$$1.234567E+03"), SV("{:$>15.6LE}"), F(1.234567e3));
  test(SV("$1.234567E+03$$"), SV("{:$^15.6LE}"), F(1.234567e3));
  test(SV("0001.234567E+03"), SV("{:015.6LE}"), F(1.234567e3));
  test(SV("-1.234567E+03$$$"), SV("{:$<16.6LE}"), F(-1.234567e3));
  test(SV("$$$-1.234567E+03"), SV("{:$>16.6LE}"), F(-1.234567e3));
  test(SV("$-1.234567E+03$$"), SV("{:$^16.6LE}"), F(-1.234567e3));
  test(SV("-0001.234567E+03"), SV("{:016.6LE}"), F(-1.234567e3));

  std::locale::global(loc);
  test(SV("1#234567E+03$$$"), SV("{:$<15.6LE}"), F(1.234567e3));
  test(SV("$$$1#234567E+03"), SV("{:$>15.6LE}"), F(1.234567e3));
  test(SV("$1#234567E+03$$"), SV("{:$^15.6LE}"), F(1.234567e3));
  test(SV("0001#234567E+03"), SV("{:015.6LE}"), F(1.234567e3));
  test(SV("-1#234567E+03$$$"), SV("{:$<16.6LE}"), F(-1.234567e3));
  test(SV("$$$-1#234567E+03"), SV("{:$>16.6LE}"), F(-1.234567e3));
  test(SV("$-1#234567E+03$$"), SV("{:$^16.6LE}"), F(-1.234567e3));
  test(SV("-0001#234567E+03"), SV("{:016.6LE}"), F(-1.234567e3));

  test(SV("1.234567E+03$$$"), en_US, SV("{:$<15.6LE}"), F(1.234567e3));
  test(SV("$$$1.234567E+03"), en_US, SV("{:$>15.6LE}"), F(1.234567e3));
  test(SV("$1.234567E+03$$"), en_US, SV("{:$^15.6LE}"), F(1.234567e3));
  test(SV("0001.234567E+03"), en_US, SV("{:015.6LE}"), F(1.234567e3));
  test(SV("-1.234567E+03$$$"), en_US, SV("{:$<16.6LE}"), F(-1.234567e3));
  test(SV("$$$-1.234567E+03"), en_US, SV("{:$>16.6LE}"), F(-1.234567e3));
  test(SV("$-1.234567E+03$$"), en_US, SV("{:$^16.6LE}"), F(-1.234567e3));
  test(SV("-0001.234567E+03"), en_US, SV("{:016.6LE}"), F(-1.234567e3));

  std::locale::global(en_US);
  test(SV("1#234567E+03$$$"), loc, SV("{:$<15.6LE}"), F(1.234567e3));
  test(SV("$$$1#234567E+03"), loc, SV("{:$>15.6LE}"), F(1.234567e3));
  test(SV("$1#234567E+03$$"), loc, SV("{:$^15.6LE}"), F(1.234567e3));
  test(SV("0001#234567E+03"), loc, SV("{:015.6LE}"), F(1.234567e3));
  test(SV("-1#234567E+03$$$"), loc, SV("{:$<16.6LE}"), F(-1.234567e3));
  test(SV("$$$-1#234567E+03"), loc, SV("{:$>16.6LE}"), F(-1.234567e3));
  test(SV("$-1#234567E+03$$"), loc, SV("{:$^16.6LE}"), F(-1.234567e3));
  test(SV("-0001#234567E+03"), loc, SV("{:016.6LE}"), F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_fixed_lower_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(SV("0.000001"), SV("{:.6Lf}"), F(1.234567e-6));
  test(SV("0.000012"), SV("{:.6Lf}"), F(1.234567e-5));
  test(SV("0.000123"), SV("{:.6Lf}"), F(1.234567e-4));
  test(SV("0.001235"), SV("{:.6Lf}"), F(1.234567e-3));
  test(SV("0.012346"), SV("{:.6Lf}"), F(1.234567e-2));
  test(SV("0.123457"), SV("{:.6Lf}"), F(1.234567e-1));
  test(SV("1.234567"), SV("{:.6Lf}"), F(1.234567e0));
  test(SV("12.345670"), SV("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("123.456700"), SV("{:.6Lf}"), F(1.234567e2));
    test(SV("1,234.567000"), SV("{:.6Lf}"), F(1.234567e3));
    test(SV("12,345.670000"), SV("{:.6Lf}"), F(1.234567e4));
    test(SV("123,456.700000"), SV("{:.6Lf}"), F(1.234567e5));
    test(SV("1,234,567.000000"), SV("{:.6Lf}"), F(1.234567e6));
    test(SV("12,345,670.000000"), SV("{:.6Lf}"), F(1.234567e7));
    test(SV("123,456,700,000,000,000,000.000000"), SV("{:.6Lf}"), F(1.234567e20));
  }
  test(SV("-0.000001"), SV("{:.6Lf}"), F(-1.234567e-6));
  test(SV("-0.000012"), SV("{:.6Lf}"), F(-1.234567e-5));
  test(SV("-0.000123"), SV("{:.6Lf}"), F(-1.234567e-4));
  test(SV("-0.001235"), SV("{:.6Lf}"), F(-1.234567e-3));
  test(SV("-0.012346"), SV("{:.6Lf}"), F(-1.234567e-2));
  test(SV("-0.123457"), SV("{:.6Lf}"), F(-1.234567e-1));
  test(SV("-1.234567"), SV("{:.6Lf}"), F(-1.234567e0));
  test(SV("-12.345670"), SV("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("-123.456700"), SV("{:.6Lf}"), F(-1.234567e2));
    test(SV("-1,234.567000"), SV("{:.6Lf}"), F(-1.234567e3));
    test(SV("-12,345.670000"), SV("{:.6Lf}"), F(-1.234567e4));
    test(SV("-123,456.700000"), SV("{:.6Lf}"), F(-1.234567e5));
    test(SV("-1,234,567.000000"), SV("{:.6Lf}"), F(-1.234567e6));
    test(SV("-12,345,670.000000"), SV("{:.6Lf}"), F(-1.234567e7));
    test(SV("-123,456,700,000,000,000,000.000000"), SV("{:.6Lf}"), F(-1.234567e20));
  }

  std::locale::global(loc);
  test(SV("0#000001"), SV("{:.6Lf}"), F(1.234567e-6));
  test(SV("0#000012"), SV("{:.6Lf}"), F(1.234567e-5));
  test(SV("0#000123"), SV("{:.6Lf}"), F(1.234567e-4));
  test(SV("0#001235"), SV("{:.6Lf}"), F(1.234567e-3));
  test(SV("0#012346"), SV("{:.6Lf}"), F(1.234567e-2));
  test(SV("0#123457"), SV("{:.6Lf}"), F(1.234567e-1));
  test(SV("1#234567"), SV("{:.6Lf}"), F(1.234567e0));
  test(SV("1_2#345670"), SV("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("12_3#456700"), SV("{:.6Lf}"), F(1.234567e2));
    test(SV("1_23_4#567000"), SV("{:.6Lf}"), F(1.234567e3));
    test(SV("12_34_5#670000"), SV("{:.6Lf}"), F(1.234567e4));
    test(SV("123_45_6#700000"), SV("{:.6Lf}"), F(1.234567e5));
    test(SV("1_234_56_7#000000"), SV("{:.6Lf}"), F(1.234567e6));
    test(SV("12_345_67_0#000000"), SV("{:.6Lf}"), F(1.234567e7));
    test(SV("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), SV("{:.6Lf}"), F(1.234567e20));
  }
  test(SV("-0#000001"), SV("{:.6Lf}"), F(-1.234567e-6));
  test(SV("-0#000012"), SV("{:.6Lf}"), F(-1.234567e-5));
  test(SV("-0#000123"), SV("{:.6Lf}"), F(-1.234567e-4));
  test(SV("-0#001235"), SV("{:.6Lf}"), F(-1.234567e-3));
  test(SV("-0#012346"), SV("{:.6Lf}"), F(-1.234567e-2));
  test(SV("-0#123457"), SV("{:.6Lf}"), F(-1.234567e-1));
  test(SV("-1#234567"), SV("{:.6Lf}"), F(-1.234567e0));
  test(SV("-1_2#345670"), SV("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("-12_3#456700"), SV("{:.6Lf}"), F(-1.234567e2));
    test(SV("-1_23_4#567000"), SV("{:.6Lf}"), F(-1.234567e3));
    test(SV("-12_34_5#670000"), SV("{:.6Lf}"), F(-1.234567e4));
    test(SV("-123_45_6#700000"), SV("{:.6Lf}"), F(-1.234567e5));
    test(SV("-1_234_56_7#000000"), SV("{:.6Lf}"), F(-1.234567e6));
    test(SV("-12_345_67_0#000000"), SV("{:.6Lf}"), F(-1.234567e7));
    test(SV("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), SV("{:.6Lf}"), F(-1.234567e20));
  }

  test(SV("0.000001"), en_US, SV("{:.6Lf}"), F(1.234567e-6));
  test(SV("0.000012"), en_US, SV("{:.6Lf}"), F(1.234567e-5));
  test(SV("0.000123"), en_US, SV("{:.6Lf}"), F(1.234567e-4));
  test(SV("0.001235"), en_US, SV("{:.6Lf}"), F(1.234567e-3));
  test(SV("0.012346"), en_US, SV("{:.6Lf}"), F(1.234567e-2));
  test(SV("0.123457"), en_US, SV("{:.6Lf}"), F(1.234567e-1));
  test(SV("1.234567"), en_US, SV("{:.6Lf}"), F(1.234567e0));
  test(SV("12.345670"), en_US, SV("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("123.456700"), en_US, SV("{:.6Lf}"), F(1.234567e2));
    test(SV("1,234.567000"), en_US, SV("{:.6Lf}"), F(1.234567e3));
    test(SV("12,345.670000"), en_US, SV("{:.6Lf}"), F(1.234567e4));
    test(SV("123,456.700000"), en_US, SV("{:.6Lf}"), F(1.234567e5));
    test(SV("1,234,567.000000"), en_US, SV("{:.6Lf}"), F(1.234567e6));
    test(SV("12,345,670.000000"), en_US, SV("{:.6Lf}"), F(1.234567e7));
    test(SV("123,456,700,000,000,000,000.000000"), en_US, SV("{:.6Lf}"), F(1.234567e20));
  }
  test(SV("-0.000001"), en_US, SV("{:.6Lf}"), F(-1.234567e-6));
  test(SV("-0.000012"), en_US, SV("{:.6Lf}"), F(-1.234567e-5));
  test(SV("-0.000123"), en_US, SV("{:.6Lf}"), F(-1.234567e-4));
  test(SV("-0.001235"), en_US, SV("{:.6Lf}"), F(-1.234567e-3));
  test(SV("-0.012346"), en_US, SV("{:.6Lf}"), F(-1.234567e-2));
  test(SV("-0.123457"), en_US, SV("{:.6Lf}"), F(-1.234567e-1));
  test(SV("-1.234567"), en_US, SV("{:.6Lf}"), F(-1.234567e0));
  test(SV("-12.345670"), en_US, SV("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("-123.456700"), en_US, SV("{:.6Lf}"), F(-1.234567e2));
    test(SV("-1,234.567000"), en_US, SV("{:.6Lf}"), F(-1.234567e3));
    test(SV("-12,345.670000"), en_US, SV("{:.6Lf}"), F(-1.234567e4));
    test(SV("-123,456.700000"), en_US, SV("{:.6Lf}"), F(-1.234567e5));
    test(SV("-1,234,567.000000"), en_US, SV("{:.6Lf}"), F(-1.234567e6));
    test(SV("-12,345,670.000000"), en_US, SV("{:.6Lf}"), F(-1.234567e7));
    test(SV("-123,456,700,000,000,000,000.000000"), en_US, SV("{:.6Lf}"), F(-1.234567e20));
  }

  std::locale::global(en_US);
  test(SV("0#000001"), loc, SV("{:.6Lf}"), F(1.234567e-6));
  test(SV("0#000012"), loc, SV("{:.6Lf}"), F(1.234567e-5));
  test(SV("0#000123"), loc, SV("{:.6Lf}"), F(1.234567e-4));
  test(SV("0#001235"), loc, SV("{:.6Lf}"), F(1.234567e-3));
  test(SV("0#012346"), loc, SV("{:.6Lf}"), F(1.234567e-2));
  test(SV("0#123457"), loc, SV("{:.6Lf}"), F(1.234567e-1));
  test(SV("1#234567"), loc, SV("{:.6Lf}"), F(1.234567e0));
  test(SV("1_2#345670"), loc, SV("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("12_3#456700"), loc, SV("{:.6Lf}"), F(1.234567e2));
    test(SV("1_23_4#567000"), loc, SV("{:.6Lf}"), F(1.234567e3));
    test(SV("12_34_5#670000"), loc, SV("{:.6Lf}"), F(1.234567e4));
    test(SV("123_45_6#700000"), loc, SV("{:.6Lf}"), F(1.234567e5));
    test(SV("1_234_56_7#000000"), loc, SV("{:.6Lf}"), F(1.234567e6));
    test(SV("12_345_67_0#000000"), loc, SV("{:.6Lf}"), F(1.234567e7));
    test(SV("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), loc, SV("{:.6Lf}"), F(1.234567e20));
  }
  test(SV("-0#000001"), loc, SV("{:.6Lf}"), F(-1.234567e-6));
  test(SV("-0#000012"), loc, SV("{:.6Lf}"), F(-1.234567e-5));
  test(SV("-0#000123"), loc, SV("{:.6Lf}"), F(-1.234567e-4));
  test(SV("-0#001235"), loc, SV("{:.6Lf}"), F(-1.234567e-3));
  test(SV("-0#012346"), loc, SV("{:.6Lf}"), F(-1.234567e-2));
  test(SV("-0#123457"), loc, SV("{:.6Lf}"), F(-1.234567e-1));
  test(SV("-1#234567"), loc, SV("{:.6Lf}"), F(-1.234567e0));
  test(SV("-1_2#345670"), loc, SV("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("-12_3#456700"), loc, SV("{:.6Lf}"), F(-1.234567e2));
    test(SV("-1_23_4#567000"), loc, SV("{:.6Lf}"), F(-1.234567e3));
    test(SV("-12_34_5#670000"), loc, SV("{:.6Lf}"), F(-1.234567e4));
    test(SV("-123_45_6#700000"), loc, SV("{:.6Lf}"), F(-1.234567e5));
    test(SV("-1_234_56_7#000000"), loc, SV("{:.6Lf}"), F(-1.234567e6));
    test(SV("-12_345_67_0#000000"), loc, SV("{:.6Lf}"), F(-1.234567e7));
    test(SV("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), loc, SV("{:.6Lf}"), F(-1.234567e20));
  }

  // *** Fill, align, zero padding ***
  if constexpr (sizeof(F) > sizeof(float)) {
    std::locale::global(en_US);
    test(SV("1,234.567000$$$"), SV("{:$<15.6Lf}"), F(1.234567e3));
    test(SV("$$$1,234.567000"), SV("{:$>15.6Lf}"), F(1.234567e3));
    test(SV("$1,234.567000$$"), SV("{:$^15.6Lf}"), F(1.234567e3));
    test(SV("0001,234.567000"), SV("{:015.6Lf}"), F(1.234567e3));
    test(SV("-1,234.567000$$$"), SV("{:$<16.6Lf}"), F(-1.234567e3));
    test(SV("$$$-1,234.567000"), SV("{:$>16.6Lf}"), F(-1.234567e3));
    test(SV("$-1,234.567000$$"), SV("{:$^16.6Lf}"), F(-1.234567e3));
    test(SV("-0001,234.567000"), SV("{:016.6Lf}"), F(-1.234567e3));

    std::locale::global(loc);
    test(SV("1_23_4#567000$$$"), SV("{:$<16.6Lf}"), F(1.234567e3));
    test(SV("$$$1_23_4#567000"), SV("{:$>16.6Lf}"), F(1.234567e3));
    test(SV("$1_23_4#567000$$"), SV("{:$^16.6Lf}"), F(1.234567e3));
    test(SV("0001_23_4#567000"), SV("{:016.6Lf}"), F(1.234567e3));
    test(SV("-1_23_4#567000$$$"), SV("{:$<17.6Lf}"), F(-1.234567e3));
    test(SV("$$$-1_23_4#567000"), SV("{:$>17.6Lf}"), F(-1.234567e3));
    test(SV("$-1_23_4#567000$$"), SV("{:$^17.6Lf}"), F(-1.234567e3));
    test(SV("-0001_23_4#567000"), SV("{:017.6Lf}"), F(-1.234567e3));

    test(SV("1,234.567000$$$"), en_US, SV("{:$<15.6Lf}"), F(1.234567e3));
    test(SV("$$$1,234.567000"), en_US, SV("{:$>15.6Lf}"), F(1.234567e3));
    test(SV("$1,234.567000$$"), en_US, SV("{:$^15.6Lf}"), F(1.234567e3));
    test(SV("0001,234.567000"), en_US, SV("{:015.6Lf}"), F(1.234567e3));
    test(SV("-1,234.567000$$$"), en_US, SV("{:$<16.6Lf}"), F(-1.234567e3));
    test(SV("$$$-1,234.567000"), en_US, SV("{:$>16.6Lf}"), F(-1.234567e3));
    test(SV("$-1,234.567000$$"), en_US, SV("{:$^16.6Lf}"), F(-1.234567e3));
    test(SV("-0001,234.567000"), en_US, SV("{:016.6Lf}"), F(-1.234567e3));

    std::locale::global(en_US);
    test(SV("1_23_4#567000$$$"), loc, SV("{:$<16.6Lf}"), F(1.234567e3));
    test(SV("$$$1_23_4#567000"), loc, SV("{:$>16.6Lf}"), F(1.234567e3));
    test(SV("$1_23_4#567000$$"), loc, SV("{:$^16.6Lf}"), F(1.234567e3));
    test(SV("0001_23_4#567000"), loc, SV("{:016.6Lf}"), F(1.234567e3));
    test(SV("-1_23_4#567000$$$"), loc, SV("{:$<17.6Lf}"), F(-1.234567e3));
    test(SV("$$$-1_23_4#567000"), loc, SV("{:$>17.6Lf}"), F(-1.234567e3));
    test(SV("$-1_23_4#567000$$"), loc, SV("{:$^17.6Lf}"), F(-1.234567e3));
    test(SV("-0001_23_4#567000"), loc, SV("{:017.6Lf}"), F(-1.234567e3));
  }
}

template <class F, class CharT>
void test_floating_point_fixed_upper_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(SV("0.000001"), SV("{:.6Lf}"), F(1.234567e-6));
  test(SV("0.000012"), SV("{:.6Lf}"), F(1.234567e-5));
  test(SV("0.000123"), SV("{:.6Lf}"), F(1.234567e-4));
  test(SV("0.001235"), SV("{:.6Lf}"), F(1.234567e-3));
  test(SV("0.012346"), SV("{:.6Lf}"), F(1.234567e-2));
  test(SV("0.123457"), SV("{:.6Lf}"), F(1.234567e-1));
  test(SV("1.234567"), SV("{:.6Lf}"), F(1.234567e0));
  test(SV("12.345670"), SV("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("123.456700"), SV("{:.6Lf}"), F(1.234567e2));
    test(SV("1,234.567000"), SV("{:.6Lf}"), F(1.234567e3));
    test(SV("12,345.670000"), SV("{:.6Lf}"), F(1.234567e4));
    test(SV("123,456.700000"), SV("{:.6Lf}"), F(1.234567e5));
    test(SV("1,234,567.000000"), SV("{:.6Lf}"), F(1.234567e6));
    test(SV("12,345,670.000000"), SV("{:.6Lf}"), F(1.234567e7));
    test(SV("123,456,700,000,000,000,000.000000"), SV("{:.6Lf}"), F(1.234567e20));
  }
  test(SV("-0.000001"), SV("{:.6Lf}"), F(-1.234567e-6));
  test(SV("-0.000012"), SV("{:.6Lf}"), F(-1.234567e-5));
  test(SV("-0.000123"), SV("{:.6Lf}"), F(-1.234567e-4));
  test(SV("-0.001235"), SV("{:.6Lf}"), F(-1.234567e-3));
  test(SV("-0.012346"), SV("{:.6Lf}"), F(-1.234567e-2));
  test(SV("-0.123457"), SV("{:.6Lf}"), F(-1.234567e-1));
  test(SV("-1.234567"), SV("{:.6Lf}"), F(-1.234567e0));
  test(SV("-12.345670"), SV("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("-123.456700"), SV("{:.6Lf}"), F(-1.234567e2));
    test(SV("-1,234.567000"), SV("{:.6Lf}"), F(-1.234567e3));
    test(SV("-12,345.670000"), SV("{:.6Lf}"), F(-1.234567e4));
    test(SV("-123,456.700000"), SV("{:.6Lf}"), F(-1.234567e5));
    test(SV("-1,234,567.000000"), SV("{:.6Lf}"), F(-1.234567e6));
    test(SV("-12,345,670.000000"), SV("{:.6Lf}"), F(-1.234567e7));
    test(SV("-123,456,700,000,000,000,000.000000"), SV("{:.6Lf}"), F(-1.234567e20));
  }

  std::locale::global(loc);
  test(SV("0#000001"), SV("{:.6Lf}"), F(1.234567e-6));
  test(SV("0#000012"), SV("{:.6Lf}"), F(1.234567e-5));
  test(SV("0#000123"), SV("{:.6Lf}"), F(1.234567e-4));
  test(SV("0#001235"), SV("{:.6Lf}"), F(1.234567e-3));
  test(SV("0#012346"), SV("{:.6Lf}"), F(1.234567e-2));
  test(SV("0#123457"), SV("{:.6Lf}"), F(1.234567e-1));
  test(SV("1#234567"), SV("{:.6Lf}"), F(1.234567e0));
  test(SV("1_2#345670"), SV("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("12_3#456700"), SV("{:.6Lf}"), F(1.234567e2));
    test(SV("1_23_4#567000"), SV("{:.6Lf}"), F(1.234567e3));
    test(SV("12_34_5#670000"), SV("{:.6Lf}"), F(1.234567e4));
    test(SV("123_45_6#700000"), SV("{:.6Lf}"), F(1.234567e5));
    test(SV("1_234_56_7#000000"), SV("{:.6Lf}"), F(1.234567e6));
    test(SV("12_345_67_0#000000"), SV("{:.6Lf}"), F(1.234567e7));
    test(SV("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), SV("{:.6Lf}"), F(1.234567e20));
  }
  test(SV("-0#000001"), SV("{:.6Lf}"), F(-1.234567e-6));
  test(SV("-0#000012"), SV("{:.6Lf}"), F(-1.234567e-5));
  test(SV("-0#000123"), SV("{:.6Lf}"), F(-1.234567e-4));
  test(SV("-0#001235"), SV("{:.6Lf}"), F(-1.234567e-3));
  test(SV("-0#012346"), SV("{:.6Lf}"), F(-1.234567e-2));
  test(SV("-0#123457"), SV("{:.6Lf}"), F(-1.234567e-1));
  test(SV("-1#234567"), SV("{:.6Lf}"), F(-1.234567e0));
  test(SV("-1_2#345670"), SV("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("-12_3#456700"), SV("{:.6Lf}"), F(-1.234567e2));
    test(SV("-1_23_4#567000"), SV("{:.6Lf}"), F(-1.234567e3));
    test(SV("-12_34_5#670000"), SV("{:.6Lf}"), F(-1.234567e4));
    test(SV("-123_45_6#700000"), SV("{:.6Lf}"), F(-1.234567e5));
    test(SV("-1_234_56_7#000000"), SV("{:.6Lf}"), F(-1.234567e6));
    test(SV("-12_345_67_0#000000"), SV("{:.6Lf}"), F(-1.234567e7));
    test(SV("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), SV("{:.6Lf}"), F(-1.234567e20));
  }

  test(SV("0.000001"), en_US, SV("{:.6Lf}"), F(1.234567e-6));
  test(SV("0.000012"), en_US, SV("{:.6Lf}"), F(1.234567e-5));
  test(SV("0.000123"), en_US, SV("{:.6Lf}"), F(1.234567e-4));
  test(SV("0.001235"), en_US, SV("{:.6Lf}"), F(1.234567e-3));
  test(SV("0.012346"), en_US, SV("{:.6Lf}"), F(1.234567e-2));
  test(SV("0.123457"), en_US, SV("{:.6Lf}"), F(1.234567e-1));
  test(SV("1.234567"), en_US, SV("{:.6Lf}"), F(1.234567e0));
  test(SV("12.345670"), en_US, SV("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("123.456700"), en_US, SV("{:.6Lf}"), F(1.234567e2));
    test(SV("1,234.567000"), en_US, SV("{:.6Lf}"), F(1.234567e3));
    test(SV("12,345.670000"), en_US, SV("{:.6Lf}"), F(1.234567e4));
    test(SV("123,456.700000"), en_US, SV("{:.6Lf}"), F(1.234567e5));
    test(SV("1,234,567.000000"), en_US, SV("{:.6Lf}"), F(1.234567e6));
    test(SV("12,345,670.000000"), en_US, SV("{:.6Lf}"), F(1.234567e7));
    test(SV("123,456,700,000,000,000,000.000000"), en_US, SV("{:.6Lf}"), F(1.234567e20));
  }
  test(SV("-0.000001"), en_US, SV("{:.6Lf}"), F(-1.234567e-6));
  test(SV("-0.000012"), en_US, SV("{:.6Lf}"), F(-1.234567e-5));
  test(SV("-0.000123"), en_US, SV("{:.6Lf}"), F(-1.234567e-4));
  test(SV("-0.001235"), en_US, SV("{:.6Lf}"), F(-1.234567e-3));
  test(SV("-0.012346"), en_US, SV("{:.6Lf}"), F(-1.234567e-2));
  test(SV("-0.123457"), en_US, SV("{:.6Lf}"), F(-1.234567e-1));
  test(SV("-1.234567"), en_US, SV("{:.6Lf}"), F(-1.234567e0));
  test(SV("-12.345670"), en_US, SV("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("-123.456700"), en_US, SV("{:.6Lf}"), F(-1.234567e2));
    test(SV("-1,234.567000"), en_US, SV("{:.6Lf}"), F(-1.234567e3));
    test(SV("-12,345.670000"), en_US, SV("{:.6Lf}"), F(-1.234567e4));
    test(SV("-123,456.700000"), en_US, SV("{:.6Lf}"), F(-1.234567e5));
    test(SV("-1,234,567.000000"), en_US, SV("{:.6Lf}"), F(-1.234567e6));
    test(SV("-12,345,670.000000"), en_US, SV("{:.6Lf}"), F(-1.234567e7));
    test(SV("-123,456,700,000,000,000,000.000000"), en_US, SV("{:.6Lf}"), F(-1.234567e20));
  }

  std::locale::global(en_US);
  test(SV("0#000001"), loc, SV("{:.6Lf}"), F(1.234567e-6));
  test(SV("0#000012"), loc, SV("{:.6Lf}"), F(1.234567e-5));
  test(SV("0#000123"), loc, SV("{:.6Lf}"), F(1.234567e-4));
  test(SV("0#001235"), loc, SV("{:.6Lf}"), F(1.234567e-3));
  test(SV("0#012346"), loc, SV("{:.6Lf}"), F(1.234567e-2));
  test(SV("0#123457"), loc, SV("{:.6Lf}"), F(1.234567e-1));
  test(SV("1#234567"), loc, SV("{:.6Lf}"), F(1.234567e0));
  test(SV("1_2#345670"), loc, SV("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("12_3#456700"), loc, SV("{:.6Lf}"), F(1.234567e2));
    test(SV("1_23_4#567000"), loc, SV("{:.6Lf}"), F(1.234567e3));
    test(SV("12_34_5#670000"), loc, SV("{:.6Lf}"), F(1.234567e4));
    test(SV("123_45_6#700000"), loc, SV("{:.6Lf}"), F(1.234567e5));
    test(SV("1_234_56_7#000000"), loc, SV("{:.6Lf}"), F(1.234567e6));
    test(SV("12_345_67_0#000000"), loc, SV("{:.6Lf}"), F(1.234567e7));
    test(SV("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), loc, SV("{:.6Lf}"), F(1.234567e20));
  }
  test(SV("-0#000001"), loc, SV("{:.6Lf}"), F(-1.234567e-6));
  test(SV("-0#000012"), loc, SV("{:.6Lf}"), F(-1.234567e-5));
  test(SV("-0#000123"), loc, SV("{:.6Lf}"), F(-1.234567e-4));
  test(SV("-0#001235"), loc, SV("{:.6Lf}"), F(-1.234567e-3));
  test(SV("-0#012346"), loc, SV("{:.6Lf}"), F(-1.234567e-2));
  test(SV("-0#123457"), loc, SV("{:.6Lf}"), F(-1.234567e-1));
  test(SV("-1#234567"), loc, SV("{:.6Lf}"), F(-1.234567e0));
  test(SV("-1_2#345670"), loc, SV("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("-12_3#456700"), loc, SV("{:.6Lf}"), F(-1.234567e2));
    test(SV("-1_23_4#567000"), loc, SV("{:.6Lf}"), F(-1.234567e3));
    test(SV("-12_34_5#670000"), loc, SV("{:.6Lf}"), F(-1.234567e4));
    test(SV("-123_45_6#700000"), loc, SV("{:.6Lf}"), F(-1.234567e5));
    test(SV("-1_234_56_7#000000"), loc, SV("{:.6Lf}"), F(-1.234567e6));
    test(SV("-12_345_67_0#000000"), loc, SV("{:.6Lf}"), F(-1.234567e7));
    test(SV("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), loc, SV("{:.6Lf}"), F(-1.234567e20));
  }

  // *** Fill, align, zero padding ***
  if constexpr (sizeof(F) > sizeof(float)) {
    std::locale::global(en_US);
    test(SV("1,234.567000$$$"), SV("{:$<15.6Lf}"), F(1.234567e3));
    test(SV("$$$1,234.567000"), SV("{:$>15.6Lf}"), F(1.234567e3));
    test(SV("$1,234.567000$$"), SV("{:$^15.6Lf}"), F(1.234567e3));
    test(SV("0001,234.567000"), SV("{:015.6Lf}"), F(1.234567e3));
    test(SV("-1,234.567000$$$"), SV("{:$<16.6Lf}"), F(-1.234567e3));
    test(SV("$$$-1,234.567000"), SV("{:$>16.6Lf}"), F(-1.234567e3));
    test(SV("$-1,234.567000$$"), SV("{:$^16.6Lf}"), F(-1.234567e3));
    test(SV("-0001,234.567000"), SV("{:016.6Lf}"), F(-1.234567e3));

    std::locale::global(loc);
    test(SV("1_23_4#567000$$$"), SV("{:$<16.6Lf}"), F(1.234567e3));
    test(SV("$$$1_23_4#567000"), SV("{:$>16.6Lf}"), F(1.234567e3));
    test(SV("$1_23_4#567000$$"), SV("{:$^16.6Lf}"), F(1.234567e3));
    test(SV("0001_23_4#567000"), SV("{:016.6Lf}"), F(1.234567e3));
    test(SV("-1_23_4#567000$$$"), SV("{:$<17.6Lf}"), F(-1.234567e3));
    test(SV("$$$-1_23_4#567000"), SV("{:$>17.6Lf}"), F(-1.234567e3));
    test(SV("$-1_23_4#567000$$"), SV("{:$^17.6Lf}"), F(-1.234567e3));
    test(SV("-0001_23_4#567000"), SV("{:017.6Lf}"), F(-1.234567e3));

    test(SV("1,234.567000$$$"), en_US, SV("{:$<15.6Lf}"), F(1.234567e3));
    test(SV("$$$1,234.567000"), en_US, SV("{:$>15.6Lf}"), F(1.234567e3));
    test(SV("$1,234.567000$$"), en_US, SV("{:$^15.6Lf}"), F(1.234567e3));
    test(SV("0001,234.567000"), en_US, SV("{:015.6Lf}"), F(1.234567e3));
    test(SV("-1,234.567000$$$"), en_US, SV("{:$<16.6Lf}"), F(-1.234567e3));
    test(SV("$$$-1,234.567000"), en_US, SV("{:$>16.6Lf}"), F(-1.234567e3));
    test(SV("$-1,234.567000$$"), en_US, SV("{:$^16.6Lf}"), F(-1.234567e3));
    test(SV("-0001,234.567000"), en_US, SV("{:016.6Lf}"), F(-1.234567e3));

    std::locale::global(en_US);
    test(SV("1_23_4#567000$$$"), loc, SV("{:$<16.6Lf}"), F(1.234567e3));
    test(SV("$$$1_23_4#567000"), loc, SV("{:$>16.6Lf}"), F(1.234567e3));
    test(SV("$1_23_4#567000$$"), loc, SV("{:$^16.6Lf}"), F(1.234567e3));
    test(SV("0001_23_4#567000"), loc, SV("{:016.6Lf}"), F(1.234567e3));
    test(SV("-1_23_4#567000$$$"), loc, SV("{:$<17.6Lf}"), F(-1.234567e3));
    test(SV("$$$-1_23_4#567000"), loc, SV("{:$>17.6Lf}"), F(-1.234567e3));
    test(SV("$-1_23_4#567000$$"), loc, SV("{:$^17.6Lf}"), F(-1.234567e3));
    test(SV("-0001_23_4#567000"), loc, SV("{:017.6Lf}"), F(-1.234567e3));
  }
}

template <class F, class CharT>
void test_floating_point_general_lower_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(SV("1.23457e-06"), SV("{:.6Lg}"), F(1.234567e-6));
  test(SV("1.23457e-05"), SV("{:.6Lg}"), F(1.234567e-5));
  test(SV("0.000123457"), SV("{:.6Lg}"), F(1.234567e-4));
  test(SV("0.00123457"), SV("{:.6Lg}"), F(1.234567e-3));
  test(SV("0.0123457"), SV("{:.6Lg}"), F(1.234567e-2));
  test(SV("0.123457"), SV("{:.6Lg}"), F(1.234567e-1));
  test(SV("1.23457"), SV("{:.6Lg}"), F(1.234567e0));
  test(SV("12.3457"), SV("{:.6Lg}"), F(1.234567e1));
  test(SV("123.457"), SV("{:.6Lg}"), F(1.234567e2));
  test(SV("1,234.57"), SV("{:.6Lg}"), F(1.234567e3));
  test(SV("12,345.7"), SV("{:.6Lg}"), F(1.234567e4));
  test(SV("123,457"), SV("{:.6Lg}"), F(1.234567e5));
  test(SV("1.23457e+06"), SV("{:.6Lg}"), F(1.234567e6));
  test(SV("1.23457e+07"), SV("{:.6Lg}"), F(1.234567e7));
  test(SV("-1.23457e-06"), SV("{:.6Lg}"), F(-1.234567e-6));
  test(SV("-1.23457e-05"), SV("{:.6Lg}"), F(-1.234567e-5));
  test(SV("-0.000123457"), SV("{:.6Lg}"), F(-1.234567e-4));
  test(SV("-0.00123457"), SV("{:.6Lg}"), F(-1.234567e-3));
  test(SV("-0.0123457"), SV("{:.6Lg}"), F(-1.234567e-2));
  test(SV("-0.123457"), SV("{:.6Lg}"), F(-1.234567e-1));
  test(SV("-1.23457"), SV("{:.6Lg}"), F(-1.234567e0));
  test(SV("-12.3457"), SV("{:.6Lg}"), F(-1.234567e1));
  test(SV("-123.457"), SV("{:.6Lg}"), F(-1.234567e2));
  test(SV("-1,234.57"), SV("{:.6Lg}"), F(-1.234567e3));
  test(SV("-12,345.7"), SV("{:.6Lg}"), F(-1.234567e4));
  test(SV("-123,457"), SV("{:.6Lg}"), F(-1.234567e5));
  test(SV("-1.23457e+06"), SV("{:.6Lg}"), F(-1.234567e6));
  test(SV("-1.23457e+07"), SV("{:.6Lg}"), F(-1.234567e7));

  std::locale::global(loc);
  test(SV("1#23457e-06"), SV("{:.6Lg}"), F(1.234567e-6));
  test(SV("1#23457e-05"), SV("{:.6Lg}"), F(1.234567e-5));
  test(SV("0#000123457"), SV("{:.6Lg}"), F(1.234567e-4));
  test(SV("0#00123457"), SV("{:.6Lg}"), F(1.234567e-3));
  test(SV("0#0123457"), SV("{:.6Lg}"), F(1.234567e-2));
  test(SV("0#123457"), SV("{:.6Lg}"), F(1.234567e-1));
  test(SV("1#23457"), SV("{:.6Lg}"), F(1.234567e0));
  test(SV("1_2#3457"), SV("{:.6Lg}"), F(1.234567e1));
  test(SV("12_3#457"), SV("{:.6Lg}"), F(1.234567e2));
  test(SV("1_23_4#57"), SV("{:.6Lg}"), F(1.234567e3));
  test(SV("12_34_5#7"), SV("{:.6Lg}"), F(1.234567e4));
  test(SV("123_45_7"), SV("{:.6Lg}"), F(1.234567e5));
  test(SV("1#23457e+06"), SV("{:.6Lg}"), F(1.234567e6));
  test(SV("1#23457e+07"), SV("{:.6Lg}"), F(1.234567e7));
  test(SV("-1#23457e-06"), SV("{:.6Lg}"), F(-1.234567e-6));
  test(SV("-1#23457e-05"), SV("{:.6Lg}"), F(-1.234567e-5));
  test(SV("-0#000123457"), SV("{:.6Lg}"), F(-1.234567e-4));
  test(SV("-0#00123457"), SV("{:.6Lg}"), F(-1.234567e-3));
  test(SV("-0#0123457"), SV("{:.6Lg}"), F(-1.234567e-2));
  test(SV("-0#123457"), SV("{:.6Lg}"), F(-1.234567e-1));
  test(SV("-1#23457"), SV("{:.6Lg}"), F(-1.234567e0));
  test(SV("-1_2#3457"), SV("{:.6Lg}"), F(-1.234567e1));
  test(SV("-12_3#457"), SV("{:.6Lg}"), F(-1.234567e2));
  test(SV("-1_23_4#57"), SV("{:.6Lg}"), F(-1.234567e3));
  test(SV("-12_34_5#7"), SV("{:.6Lg}"), F(-1.234567e4));
  test(SV("-123_45_7"), SV("{:.6Lg}"), F(-1.234567e5));
  test(SV("-1#23457e+06"), SV("{:.6Lg}"), F(-1.234567e6));
  test(SV("-1#23457e+07"), SV("{:.6Lg}"), F(-1.234567e7));

  test(SV("1.23457e-06"), en_US, SV("{:.6Lg}"), F(1.234567e-6));
  test(SV("1.23457e-05"), en_US, SV("{:.6Lg}"), F(1.234567e-5));
  test(SV("0.000123457"), en_US, SV("{:.6Lg}"), F(1.234567e-4));
  test(SV("0.00123457"), en_US, SV("{:.6Lg}"), F(1.234567e-3));
  test(SV("0.0123457"), en_US, SV("{:.6Lg}"), F(1.234567e-2));
  test(SV("0.123457"), en_US, SV("{:.6Lg}"), F(1.234567e-1));
  test(SV("1.23457"), en_US, SV("{:.6Lg}"), F(1.234567e0));
  test(SV("12.3457"), en_US, SV("{:.6Lg}"), F(1.234567e1));
  test(SV("123.457"), en_US, SV("{:.6Lg}"), F(1.234567e2));
  test(SV("1,234.57"), en_US, SV("{:.6Lg}"), F(1.234567e3));
  test(SV("12,345.7"), en_US, SV("{:.6Lg}"), F(1.234567e4));
  test(SV("123,457"), en_US, SV("{:.6Lg}"), F(1.234567e5));
  test(SV("1.23457e+06"), en_US, SV("{:.6Lg}"), F(1.234567e6));
  test(SV("1.23457e+07"), en_US, SV("{:.6Lg}"), F(1.234567e7));
  test(SV("-1.23457e-06"), en_US, SV("{:.6Lg}"), F(-1.234567e-6));
  test(SV("-1.23457e-05"), en_US, SV("{:.6Lg}"), F(-1.234567e-5));
  test(SV("-0.000123457"), en_US, SV("{:.6Lg}"), F(-1.234567e-4));
  test(SV("-0.00123457"), en_US, SV("{:.6Lg}"), F(-1.234567e-3));
  test(SV("-0.0123457"), en_US, SV("{:.6Lg}"), F(-1.234567e-2));
  test(SV("-0.123457"), en_US, SV("{:.6Lg}"), F(-1.234567e-1));
  test(SV("-1.23457"), en_US, SV("{:.6Lg}"), F(-1.234567e0));
  test(SV("-12.3457"), en_US, SV("{:.6Lg}"), F(-1.234567e1));
  test(SV("-123.457"), en_US, SV("{:.6Lg}"), F(-1.234567e2));
  test(SV("-1,234.57"), en_US, SV("{:.6Lg}"), F(-1.234567e3));
  test(SV("-12,345.7"), en_US, SV("{:.6Lg}"), F(-1.234567e4));
  test(SV("-123,457"), en_US, SV("{:.6Lg}"), F(-1.234567e5));
  test(SV("-1.23457e+06"), en_US, SV("{:.6Lg}"), F(-1.234567e6));
  test(SV("-1.23457e+07"), en_US, SV("{:.6Lg}"), F(-1.234567e7));

  std::locale::global(en_US);
  test(SV("1#23457e-06"), loc, SV("{:.6Lg}"), F(1.234567e-6));
  test(SV("1#23457e-05"), loc, SV("{:.6Lg}"), F(1.234567e-5));
  test(SV("0#000123457"), loc, SV("{:.6Lg}"), F(1.234567e-4));
  test(SV("0#00123457"), loc, SV("{:.6Lg}"), F(1.234567e-3));
  test(SV("0#0123457"), loc, SV("{:.6Lg}"), F(1.234567e-2));
  test(SV("0#123457"), loc, SV("{:.6Lg}"), F(1.234567e-1));
  test(SV("1#23457"), loc, SV("{:.6Lg}"), F(1.234567e0));
  test(SV("1_2#3457"), loc, SV("{:.6Lg}"), F(1.234567e1));
  test(SV("12_3#457"), loc, SV("{:.6Lg}"), F(1.234567e2));
  test(SV("1_23_4#57"), loc, SV("{:.6Lg}"), F(1.234567e3));
  test(SV("12_34_5#7"), loc, SV("{:.6Lg}"), F(1.234567e4));
  test(SV("123_45_7"), loc, SV("{:.6Lg}"), F(1.234567e5));
  test(SV("1#23457e+06"), loc, SV("{:.6Lg}"), F(1.234567e6));
  test(SV("1#23457e+07"), loc, SV("{:.6Lg}"), F(1.234567e7));
  test(SV("-1#23457e-06"), loc, SV("{:.6Lg}"), F(-1.234567e-6));
  test(SV("-1#23457e-05"), loc, SV("{:.6Lg}"), F(-1.234567e-5));
  test(SV("-0#000123457"), loc, SV("{:.6Lg}"), F(-1.234567e-4));
  test(SV("-0#00123457"), loc, SV("{:.6Lg}"), F(-1.234567e-3));
  test(SV("-0#0123457"), loc, SV("{:.6Lg}"), F(-1.234567e-2));
  test(SV("-0#123457"), loc, SV("{:.6Lg}"), F(-1.234567e-1));
  test(SV("-1#23457"), loc, SV("{:.6Lg}"), F(-1.234567e0));
  test(SV("-1_2#3457"), loc, SV("{:.6Lg}"), F(-1.234567e1));
  test(SV("-12_3#457"), loc, SV("{:.6Lg}"), F(-1.234567e2));
  test(SV("-1_23_4#57"), loc, SV("{:.6Lg}"), F(-1.234567e3));
  test(SV("-12_34_5#7"), loc, SV("{:.6Lg}"), F(-1.234567e4));
  test(SV("-123_45_7"), loc, SV("{:.6Lg}"), F(-1.234567e5));
  test(SV("-1#23457e+06"), loc, SV("{:.6Lg}"), F(-1.234567e6));
  test(SV("-1#23457e+07"), loc, SV("{:.6Lg}"), F(-1.234567e7));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(SV("1,234.57$$$"), SV("{:$<11.6Lg}"), F(1.234567e3));
  test(SV("$$$1,234.57"), SV("{:$>11.6Lg}"), F(1.234567e3));
  test(SV("$1,234.57$$"), SV("{:$^11.6Lg}"), F(1.234567e3));
  test(SV("0001,234.57"), SV("{:011.6Lg}"), F(1.234567e3));
  test(SV("-1,234.57$$$"), SV("{:$<12.6Lg}"), F(-1.234567e3));
  test(SV("$$$-1,234.57"), SV("{:$>12.6Lg}"), F(-1.234567e3));
  test(SV("$-1,234.57$$"), SV("{:$^12.6Lg}"), F(-1.234567e3));
  test(SV("-0001,234.57"), SV("{:012.6Lg}"), F(-1.234567e3));

  std::locale::global(loc);
  test(SV("1_23_4#57$$$"), SV("{:$<12.6Lg}"), F(1.234567e3));
  test(SV("$$$1_23_4#57"), SV("{:$>12.6Lg}"), F(1.234567e3));
  test(SV("$1_23_4#57$$"), SV("{:$^12.6Lg}"), F(1.234567e3));
  test(SV("0001_23_4#57"), SV("{:012.6Lg}"), F(1.234567e3));
  test(SV("-1_23_4#57$$$"), SV("{:$<13.6Lg}"), F(-1.234567e3));
  test(SV("$$$-1_23_4#57"), SV("{:$>13.6Lg}"), F(-1.234567e3));
  test(SV("$-1_23_4#57$$"), SV("{:$^13.6Lg}"), F(-1.234567e3));
  test(SV("-0001_23_4#57"), SV("{:013.6Lg}"), F(-1.234567e3));

  test(SV("1,234.57$$$"), en_US, SV("{:$<11.6Lg}"), F(1.234567e3));
  test(SV("$$$1,234.57"), en_US, SV("{:$>11.6Lg}"), F(1.234567e3));
  test(SV("$1,234.57$$"), en_US, SV("{:$^11.6Lg}"), F(1.234567e3));
  test(SV("0001,234.57"), en_US, SV("{:011.6Lg}"), F(1.234567e3));
  test(SV("-1,234.57$$$"), en_US, SV("{:$<12.6Lg}"), F(-1.234567e3));
  test(SV("$$$-1,234.57"), en_US, SV("{:$>12.6Lg}"), F(-1.234567e3));
  test(SV("$-1,234.57$$"), en_US, SV("{:$^12.6Lg}"), F(-1.234567e3));
  test(SV("-0001,234.57"), en_US, SV("{:012.6Lg}"), F(-1.234567e3));

  std::locale::global(en_US);
  test(SV("1_23_4#57$$$"), loc, SV("{:$<12.6Lg}"), F(1.234567e3));
  test(SV("$$$1_23_4#57"), loc, SV("{:$>12.6Lg}"), F(1.234567e3));
  test(SV("$1_23_4#57$$"), loc, SV("{:$^12.6Lg}"), F(1.234567e3));
  test(SV("0001_23_4#57"), loc, SV("{:012.6Lg}"), F(1.234567e3));
  test(SV("-1_23_4#57$$$"), loc, SV("{:$<13.6Lg}"), F(-1.234567e3));
  test(SV("$$$-1_23_4#57"), loc, SV("{:$>13.6Lg}"), F(-1.234567e3));
  test(SV("$-1_23_4#57$$"), loc, SV("{:$^13.6Lg}"), F(-1.234567e3));
  test(SV("-0001_23_4#57"), loc, SV("{:013.6Lg}"), F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_general_upper_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(SV("1.23457E-06"), SV("{:.6LG}"), F(1.234567e-6));
  test(SV("1.23457E-05"), SV("{:.6LG}"), F(1.234567e-5));
  test(SV("0.000123457"), SV("{:.6LG}"), F(1.234567e-4));
  test(SV("0.00123457"), SV("{:.6LG}"), F(1.234567e-3));
  test(SV("0.0123457"), SV("{:.6LG}"), F(1.234567e-2));
  test(SV("0.123457"), SV("{:.6LG}"), F(1.234567e-1));
  test(SV("1.23457"), SV("{:.6LG}"), F(1.234567e0));
  test(SV("12.3457"), SV("{:.6LG}"), F(1.234567e1));
  test(SV("123.457"), SV("{:.6LG}"), F(1.234567e2));
  test(SV("1,234.57"), SV("{:.6LG}"), F(1.234567e3));
  test(SV("12,345.7"), SV("{:.6LG}"), F(1.234567e4));
  test(SV("123,457"), SV("{:.6LG}"), F(1.234567e5));
  test(SV("1.23457E+06"), SV("{:.6LG}"), F(1.234567e6));
  test(SV("1.23457E+07"), SV("{:.6LG}"), F(1.234567e7));
  test(SV("-1.23457E-06"), SV("{:.6LG}"), F(-1.234567e-6));
  test(SV("-1.23457E-05"), SV("{:.6LG}"), F(-1.234567e-5));
  test(SV("-0.000123457"), SV("{:.6LG}"), F(-1.234567e-4));
  test(SV("-0.00123457"), SV("{:.6LG}"), F(-1.234567e-3));
  test(SV("-0.0123457"), SV("{:.6LG}"), F(-1.234567e-2));
  test(SV("-0.123457"), SV("{:.6LG}"), F(-1.234567e-1));
  test(SV("-1.23457"), SV("{:.6LG}"), F(-1.234567e0));
  test(SV("-12.3457"), SV("{:.6LG}"), F(-1.234567e1));
  test(SV("-123.457"), SV("{:.6LG}"), F(-1.234567e2));
  test(SV("-1,234.57"), SV("{:.6LG}"), F(-1.234567e3));
  test(SV("-12,345.7"), SV("{:.6LG}"), F(-1.234567e4));
  test(SV("-123,457"), SV("{:.6LG}"), F(-1.234567e5));
  test(SV("-1.23457E+06"), SV("{:.6LG}"), F(-1.234567e6));
  test(SV("-1.23457E+07"), SV("{:.6LG}"), F(-1.234567e7));

  std::locale::global(loc);
  test(SV("1#23457E-06"), SV("{:.6LG}"), F(1.234567e-6));
  test(SV("1#23457E-05"), SV("{:.6LG}"), F(1.234567e-5));
  test(SV("0#000123457"), SV("{:.6LG}"), F(1.234567e-4));
  test(SV("0#00123457"), SV("{:.6LG}"), F(1.234567e-3));
  test(SV("0#0123457"), SV("{:.6LG}"), F(1.234567e-2));
  test(SV("0#123457"), SV("{:.6LG}"), F(1.234567e-1));
  test(SV("1#23457"), SV("{:.6LG}"), F(1.234567e0));
  test(SV("1_2#3457"), SV("{:.6LG}"), F(1.234567e1));
  test(SV("12_3#457"), SV("{:.6LG}"), F(1.234567e2));
  test(SV("1_23_4#57"), SV("{:.6LG}"), F(1.234567e3));
  test(SV("12_34_5#7"), SV("{:.6LG}"), F(1.234567e4));
  test(SV("123_45_7"), SV("{:.6LG}"), F(1.234567e5));
  test(SV("1#23457E+06"), SV("{:.6LG}"), F(1.234567e6));
  test(SV("1#23457E+07"), SV("{:.6LG}"), F(1.234567e7));
  test(SV("-1#23457E-06"), SV("{:.6LG}"), F(-1.234567e-6));
  test(SV("-1#23457E-05"), SV("{:.6LG}"), F(-1.234567e-5));
  test(SV("-0#000123457"), SV("{:.6LG}"), F(-1.234567e-4));
  test(SV("-0#00123457"), SV("{:.6LG}"), F(-1.234567e-3));
  test(SV("-0#0123457"), SV("{:.6LG}"), F(-1.234567e-2));
  test(SV("-0#123457"), SV("{:.6LG}"), F(-1.234567e-1));
  test(SV("-1#23457"), SV("{:.6LG}"), F(-1.234567e0));
  test(SV("-1_2#3457"), SV("{:.6LG}"), F(-1.234567e1));
  test(SV("-12_3#457"), SV("{:.6LG}"), F(-1.234567e2));
  test(SV("-1_23_4#57"), SV("{:.6LG}"), F(-1.234567e3));
  test(SV("-12_34_5#7"), SV("{:.6LG}"), F(-1.234567e4));
  test(SV("-123_45_7"), SV("{:.6LG}"), F(-1.234567e5));
  test(SV("-1#23457E+06"), SV("{:.6LG}"), F(-1.234567e6));
  test(SV("-1#23457E+07"), SV("{:.6LG}"), F(-1.234567e7));

  test(SV("1.23457E-06"), en_US, SV("{:.6LG}"), F(1.234567e-6));
  test(SV("1.23457E-05"), en_US, SV("{:.6LG}"), F(1.234567e-5));
  test(SV("0.000123457"), en_US, SV("{:.6LG}"), F(1.234567e-4));
  test(SV("0.00123457"), en_US, SV("{:.6LG}"), F(1.234567e-3));
  test(SV("0.0123457"), en_US, SV("{:.6LG}"), F(1.234567e-2));
  test(SV("0.123457"), en_US, SV("{:.6LG}"), F(1.234567e-1));
  test(SV("1.23457"), en_US, SV("{:.6LG}"), F(1.234567e0));
  test(SV("12.3457"), en_US, SV("{:.6LG}"), F(1.234567e1));
  test(SV("123.457"), en_US, SV("{:.6LG}"), F(1.234567e2));
  test(SV("1,234.57"), en_US, SV("{:.6LG}"), F(1.234567e3));
  test(SV("12,345.7"), en_US, SV("{:.6LG}"), F(1.234567e4));
  test(SV("123,457"), en_US, SV("{:.6LG}"), F(1.234567e5));
  test(SV("1.23457E+06"), en_US, SV("{:.6LG}"), F(1.234567e6));
  test(SV("1.23457E+07"), en_US, SV("{:.6LG}"), F(1.234567e7));
  test(SV("-1.23457E-06"), en_US, SV("{:.6LG}"), F(-1.234567e-6));
  test(SV("-1.23457E-05"), en_US, SV("{:.6LG}"), F(-1.234567e-5));
  test(SV("-0.000123457"), en_US, SV("{:.6LG}"), F(-1.234567e-4));
  test(SV("-0.00123457"), en_US, SV("{:.6LG}"), F(-1.234567e-3));
  test(SV("-0.0123457"), en_US, SV("{:.6LG}"), F(-1.234567e-2));
  test(SV("-0.123457"), en_US, SV("{:.6LG}"), F(-1.234567e-1));
  test(SV("-1.23457"), en_US, SV("{:.6LG}"), F(-1.234567e0));
  test(SV("-12.3457"), en_US, SV("{:.6LG}"), F(-1.234567e1));
  test(SV("-123.457"), en_US, SV("{:.6LG}"), F(-1.234567e2));
  test(SV("-1,234.57"), en_US, SV("{:.6LG}"), F(-1.234567e3));
  test(SV("-12,345.7"), en_US, SV("{:.6LG}"), F(-1.234567e4));
  test(SV("-123,457"), en_US, SV("{:.6LG}"), F(-1.234567e5));
  test(SV("-1.23457E+06"), en_US, SV("{:.6LG}"), F(-1.234567e6));
  test(SV("-1.23457E+07"), en_US, SV("{:.6LG}"), F(-1.234567e7));

  std::locale::global(en_US);
  test(SV("1#23457E-06"), loc, SV("{:.6LG}"), F(1.234567e-6));
  test(SV("1#23457E-05"), loc, SV("{:.6LG}"), F(1.234567e-5));
  test(SV("0#000123457"), loc, SV("{:.6LG}"), F(1.234567e-4));
  test(SV("0#00123457"), loc, SV("{:.6LG}"), F(1.234567e-3));
  test(SV("0#0123457"), loc, SV("{:.6LG}"), F(1.234567e-2));
  test(SV("0#123457"), loc, SV("{:.6LG}"), F(1.234567e-1));
  test(SV("1#23457"), loc, SV("{:.6LG}"), F(1.234567e0));
  test(SV("1_2#3457"), loc, SV("{:.6LG}"), F(1.234567e1));
  test(SV("12_3#457"), loc, SV("{:.6LG}"), F(1.234567e2));
  test(SV("1_23_4#57"), loc, SV("{:.6LG}"), F(1.234567e3));
  test(SV("12_34_5#7"), loc, SV("{:.6LG}"), F(1.234567e4));
  test(SV("123_45_7"), loc, SV("{:.6LG}"), F(1.234567e5));
  test(SV("1#23457E+06"), loc, SV("{:.6LG}"), F(1.234567e6));
  test(SV("1#23457E+07"), loc, SV("{:.6LG}"), F(1.234567e7));
  test(SV("-1#23457E-06"), loc, SV("{:.6LG}"), F(-1.234567e-6));
  test(SV("-1#23457E-05"), loc, SV("{:.6LG}"), F(-1.234567e-5));
  test(SV("-0#000123457"), loc, SV("{:.6LG}"), F(-1.234567e-4));
  test(SV("-0#00123457"), loc, SV("{:.6LG}"), F(-1.234567e-3));
  test(SV("-0#0123457"), loc, SV("{:.6LG}"), F(-1.234567e-2));
  test(SV("-0#123457"), loc, SV("{:.6LG}"), F(-1.234567e-1));
  test(SV("-1#23457"), loc, SV("{:.6LG}"), F(-1.234567e0));
  test(SV("-1_2#3457"), loc, SV("{:.6LG}"), F(-1.234567e1));
  test(SV("-12_3#457"), loc, SV("{:.6LG}"), F(-1.234567e2));
  test(SV("-1_23_4#57"), loc, SV("{:.6LG}"), F(-1.234567e3));
  test(SV("-12_34_5#7"), loc, SV("{:.6LG}"), F(-1.234567e4));
  test(SV("-123_45_7"), loc, SV("{:.6LG}"), F(-1.234567e5));
  test(SV("-1#23457E+06"), loc, SV("{:.6LG}"), F(-1.234567e6));
  test(SV("-1#23457E+07"), loc, SV("{:.6LG}"), F(-1.234567e7));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(SV("1,234.57$$$"), SV("{:$<11.6LG}"), F(1.234567e3));
  test(SV("$$$1,234.57"), SV("{:$>11.6LG}"), F(1.234567e3));
  test(SV("$1,234.57$$"), SV("{:$^11.6LG}"), F(1.234567e3));
  test(SV("0001,234.57"), SV("{:011.6LG}"), F(1.234567e3));
  test(SV("-1,234.57$$$"), SV("{:$<12.6LG}"), F(-1.234567e3));
  test(SV("$$$-1,234.57"), SV("{:$>12.6LG}"), F(-1.234567e3));
  test(SV("$-1,234.57$$"), SV("{:$^12.6LG}"), F(-1.234567e3));
  test(SV("-0001,234.57"), SV("{:012.6LG}"), F(-1.234567e3));

  std::locale::global(loc);
  test(SV("1_23_4#57$$$"), SV("{:$<12.6LG}"), F(1.234567e3));
  test(SV("$$$1_23_4#57"), SV("{:$>12.6LG}"), F(1.234567e3));
  test(SV("$1_23_4#57$$"), SV("{:$^12.6LG}"), F(1.234567e3));
  test(SV("0001_23_4#57"), SV("{:012.6LG}"), F(1.234567e3));
  test(SV("-1_23_4#57$$$"), SV("{:$<13.6LG}"), F(-1.234567e3));
  test(SV("$$$-1_23_4#57"), SV("{:$>13.6LG}"), F(-1.234567e3));
  test(SV("$-1_23_4#57$$"), SV("{:$^13.6LG}"), F(-1.234567e3));
  test(SV("-0001_23_4#57"), SV("{:013.6LG}"), F(-1.234567e3));

  test(SV("1,234.57$$$"), en_US, SV("{:$<11.6LG}"), F(1.234567e3));
  test(SV("$$$1,234.57"), en_US, SV("{:$>11.6LG}"), F(1.234567e3));
  test(SV("$1,234.57$$"), en_US, SV("{:$^11.6LG}"), F(1.234567e3));
  test(SV("0001,234.57"), en_US, SV("{:011.6LG}"), F(1.234567e3));
  test(SV("-1,234.57$$$"), en_US, SV("{:$<12.6LG}"), F(-1.234567e3));
  test(SV("$$$-1,234.57"), en_US, SV("{:$>12.6LG}"), F(-1.234567e3));
  test(SV("$-1,234.57$$"), en_US, SV("{:$^12.6LG}"), F(-1.234567e3));
  test(SV("-0001,234.57"), en_US, SV("{:012.6LG}"), F(-1.234567e3));

  std::locale::global(en_US);
  test(SV("1_23_4#57$$$"), loc, SV("{:$<12.6LG}"), F(1.234567e3));
  test(SV("$$$1_23_4#57"), loc, SV("{:$>12.6LG}"), F(1.234567e3));
  test(SV("$1_23_4#57$$"), loc, SV("{:$^12.6LG}"), F(1.234567e3));
  test(SV("0001_23_4#57"), loc, SV("{:012.6LG}"), F(1.234567e3));
  test(SV("-1_23_4#57$$$"), loc, SV("{:$<13.6LG}"), F(-1.234567e3));
  test(SV("$$$-1_23_4#57"), loc, SV("{:$>13.6LG}"), F(-1.234567e3));
  test(SV("$-1_23_4#57$$"), loc, SV("{:$^13.6LG}"), F(-1.234567e3));
  test(SV("-0001_23_4#57"), loc, SV("{:013.6LG}"), F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_default() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(SV("1.234567e-06"), SV("{:L}"), F(1.234567e-6));
  test(SV("1.234567e-05"), SV("{:L}"), F(1.234567e-5));
  test(SV("0.0001234567"), SV("{:L}"), F(1.234567e-4));
  test(SV("0.001234567"), SV("{:L}"), F(1.234567e-3));
  test(SV("0.01234567"), SV("{:L}"), F(1.234567e-2));
  test(SV("0.1234567"), SV("{:L}"), F(1.234567e-1));
  test(SV("1.234567"), SV("{:L}"), F(1.234567e0));
  test(SV("12.34567"), SV("{:L}"), F(1.234567e1));
  test(SV("123.4567"), SV("{:L}"), F(1.234567e2));
  test(SV("1,234.567"), SV("{:L}"), F(1.234567e3));
  test(SV("12,345.67"), SV("{:L}"), F(1.234567e4));
  test(SV("123,456.7"), SV("{:L}"), F(1.234567e5));
  test(SV("1,234,567"), SV("{:L}"), F(1.234567e6));
  test(SV("12,345,670"), SV("{:L}"), F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("123,456,700"), SV("{:L}"), F(1.234567e8));
    test(SV("1,234,567,000"), SV("{:L}"), F(1.234567e9));
    test(SV("12,345,670,000"), SV("{:L}"), F(1.234567e10));
    test(SV("123,456,700,000"), SV("{:L}"), F(1.234567e11));
    test(SV("1.234567e+12"), SV("{:L}"), F(1.234567e12));
    test(SV("1.234567e+13"), SV("{:L}"), F(1.234567e13));
  }
  test(SV("-1.234567e-06"), SV("{:L}"), F(-1.234567e-6));
  test(SV("-1.234567e-05"), SV("{:L}"), F(-1.234567e-5));
  test(SV("-0.0001234567"), SV("{:L}"), F(-1.234567e-4));
  test(SV("-0.001234567"), SV("{:L}"), F(-1.234567e-3));
  test(SV("-0.01234567"), SV("{:L}"), F(-1.234567e-2));
  test(SV("-0.1234567"), SV("{:L}"), F(-1.234567e-1));
  test(SV("-1.234567"), SV("{:L}"), F(-1.234567e0));
  test(SV("-12.34567"), SV("{:L}"), F(-1.234567e1));
  test(SV("-123.4567"), SV("{:L}"), F(-1.234567e2));
  test(SV("-1,234.567"), SV("{:L}"), F(-1.234567e3));
  test(SV("-12,345.67"), SV("{:L}"), F(-1.234567e4));
  test(SV("-123,456.7"), SV("{:L}"), F(-1.234567e5));
  test(SV("-1,234,567"), SV("{:L}"), F(-1.234567e6));
  test(SV("-12,345,670"), SV("{:L}"), F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("-123,456,700"), SV("{:L}"), F(-1.234567e8));
    test(SV("-1,234,567,000"), SV("{:L}"), F(-1.234567e9));
    test(SV("-12,345,670,000"), SV("{:L}"), F(-1.234567e10));
    test(SV("-123,456,700,000"), SV("{:L}"), F(-1.234567e11));
    test(SV("-1.234567e+12"), SV("{:L}"), F(-1.234567e12));
    test(SV("-1.234567e+13"), SV("{:L}"), F(-1.234567e13));
  }

  std::locale::global(loc);
  test(SV("1#234567e-06"), SV("{:L}"), F(1.234567e-6));
  test(SV("1#234567e-05"), SV("{:L}"), F(1.234567e-5));
  test(SV("0#0001234567"), SV("{:L}"), F(1.234567e-4));
  test(SV("0#001234567"), SV("{:L}"), F(1.234567e-3));
  test(SV("0#01234567"), SV("{:L}"), F(1.234567e-2));
  test(SV("0#1234567"), SV("{:L}"), F(1.234567e-1));
  test(SV("1#234567"), SV("{:L}"), F(1.234567e0));
  test(SV("1_2#34567"), SV("{:L}"), F(1.234567e1));
  test(SV("12_3#4567"), SV("{:L}"), F(1.234567e2));
  test(SV("1_23_4#567"), SV("{:L}"), F(1.234567e3));
  test(SV("12_34_5#67"), SV("{:L}"), F(1.234567e4));
  test(SV("123_45_6#7"), SV("{:L}"), F(1.234567e5));
  test(SV("1_234_56_7"), SV("{:L}"), F(1.234567e6));
  test(SV("12_345_67_0"), SV("{:L}"), F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("1_23_456_70_0"), SV("{:L}"), F(1.234567e8));
    test(SV("1_2_34_567_00_0"), SV("{:L}"), F(1.234567e9));
    test(SV("1_2_3_45_670_00_0"), SV("{:L}"), F(1.234567e10));
    test(SV("1_2_3_4_56_700_00_0"), SV("{:L}"), F(1.234567e11));
    test(SV("1#234567e+12"), SV("{:L}"), F(1.234567e12));
    test(SV("1#234567e+13"), SV("{:L}"), F(1.234567e13));
  }
  test(SV("-1#234567e-06"), SV("{:L}"), F(-1.234567e-6));
  test(SV("-1#234567e-05"), SV("{:L}"), F(-1.234567e-5));
  test(SV("-0#0001234567"), SV("{:L}"), F(-1.234567e-4));
  test(SV("-0#001234567"), SV("{:L}"), F(-1.234567e-3));
  test(SV("-0#01234567"), SV("{:L}"), F(-1.234567e-2));
  test(SV("-0#1234567"), SV("{:L}"), F(-1.234567e-1));
  test(SV("-1#234567"), SV("{:L}"), F(-1.234567e0));
  test(SV("-1_2#34567"), SV("{:L}"), F(-1.234567e1));
  test(SV("-12_3#4567"), SV("{:L}"), F(-1.234567e2));
  test(SV("-1_23_4#567"), SV("{:L}"), F(-1.234567e3));
  test(SV("-12_34_5#67"), SV("{:L}"), F(-1.234567e4));
  test(SV("-123_45_6#7"), SV("{:L}"), F(-1.234567e5));
  test(SV("-1_234_56_7"), SV("{:L}"), F(-1.234567e6));
  test(SV("-12_345_67_0"), SV("{:L}"), F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("-1_23_456_70_0"), SV("{:L}"), F(-1.234567e8));
    test(SV("-1_2_34_567_00_0"), SV("{:L}"), F(-1.234567e9));
    test(SV("-1_2_3_45_670_00_0"), SV("{:L}"), F(-1.234567e10));
    test(SV("-1_2_3_4_56_700_00_0"), SV("{:L}"), F(-1.234567e11));
    test(SV("-1#234567e+12"), SV("{:L}"), F(-1.234567e12));
    test(SV("-1#234567e+13"), SV("{:L}"), F(-1.234567e13));
  }

  test(SV("1.234567e-06"), en_US, SV("{:L}"), F(1.234567e-6));
  test(SV("1.234567e-05"), en_US, SV("{:L}"), F(1.234567e-5));
  test(SV("0.0001234567"), en_US, SV("{:L}"), F(1.234567e-4));
  test(SV("0.001234567"), en_US, SV("{:L}"), F(1.234567e-3));
  test(SV("0.01234567"), en_US, SV("{:L}"), F(1.234567e-2));
  test(SV("0.1234567"), en_US, SV("{:L}"), F(1.234567e-1));
  test(SV("1.234567"), en_US, SV("{:L}"), F(1.234567e0));
  test(SV("12.34567"), en_US, SV("{:L}"), F(1.234567e1));
  test(SV("123.4567"), en_US, SV("{:L}"), F(1.234567e2));
  test(SV("1,234.567"), en_US, SV("{:L}"), F(1.234567e3));
  test(SV("12,345.67"), en_US, SV("{:L}"), F(1.234567e4));
  test(SV("123,456.7"), en_US, SV("{:L}"), F(1.234567e5));
  test(SV("1,234,567"), en_US, SV("{:L}"), F(1.234567e6));
  test(SV("12,345,670"), en_US, SV("{:L}"), F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("123,456,700"), en_US, SV("{:L}"), F(1.234567e8));
    test(SV("1,234,567,000"), en_US, SV("{:L}"), F(1.234567e9));
    test(SV("12,345,670,000"), en_US, SV("{:L}"), F(1.234567e10));
    test(SV("123,456,700,000"), en_US, SV("{:L}"), F(1.234567e11));
    test(SV("1.234567e+12"), en_US, SV("{:L}"), F(1.234567e12));
    test(SV("1.234567e+13"), en_US, SV("{:L}"), F(1.234567e13));
  }
  test(SV("-1.234567e-06"), en_US, SV("{:L}"), F(-1.234567e-6));
  test(SV("-1.234567e-05"), en_US, SV("{:L}"), F(-1.234567e-5));
  test(SV("-0.0001234567"), en_US, SV("{:L}"), F(-1.234567e-4));
  test(SV("-0.001234567"), en_US, SV("{:L}"), F(-1.234567e-3));
  test(SV("-0.01234567"), en_US, SV("{:L}"), F(-1.234567e-2));
  test(SV("-0.1234567"), en_US, SV("{:L}"), F(-1.234567e-1));
  test(SV("-1.234567"), en_US, SV("{:L}"), F(-1.234567e0));
  test(SV("-12.34567"), en_US, SV("{:L}"), F(-1.234567e1));
  test(SV("-123.4567"), en_US, SV("{:L}"), F(-1.234567e2));
  test(SV("-1,234.567"), en_US, SV("{:L}"), F(-1.234567e3));
  test(SV("-12,345.67"), en_US, SV("{:L}"), F(-1.234567e4));
  test(SV("-123,456.7"), en_US, SV("{:L}"), F(-1.234567e5));
  test(SV("-1,234,567"), en_US, SV("{:L}"), F(-1.234567e6));
  test(SV("-12,345,670"), en_US, SV("{:L}"), F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("-123,456,700"), en_US, SV("{:L}"), F(-1.234567e8));
    test(SV("-1,234,567,000"), en_US, SV("{:L}"), F(-1.234567e9));
    test(SV("-12,345,670,000"), en_US, SV("{:L}"), F(-1.234567e10));
    test(SV("-123,456,700,000"), en_US, SV("{:L}"), F(-1.234567e11));
    test(SV("-1.234567e+12"), en_US, SV("{:L}"), F(-1.234567e12));
    test(SV("-1.234567e+13"), en_US, SV("{:L}"), F(-1.234567e13));
  }

  std::locale::global(en_US);
  test(SV("1#234567e-06"), loc, SV("{:L}"), F(1.234567e-6));
  test(SV("1#234567e-05"), loc, SV("{:L}"), F(1.234567e-5));
  test(SV("0#0001234567"), loc, SV("{:L}"), F(1.234567e-4));
  test(SV("0#001234567"), loc, SV("{:L}"), F(1.234567e-3));
  test(SV("0#01234567"), loc, SV("{:L}"), F(1.234567e-2));
  test(SV("0#1234567"), loc, SV("{:L}"), F(1.234567e-1));
  test(SV("1#234567"), loc, SV("{:L}"), F(1.234567e0));
  test(SV("1_2#34567"), loc, SV("{:L}"), F(1.234567e1));
  test(SV("12_3#4567"), loc, SV("{:L}"), F(1.234567e2));
  test(SV("1_23_4#567"), loc, SV("{:L}"), F(1.234567e3));
  test(SV("12_34_5#67"), loc, SV("{:L}"), F(1.234567e4));
  test(SV("123_45_6#7"), loc, SV("{:L}"), F(1.234567e5));
  test(SV("1_234_56_7"), loc, SV("{:L}"), F(1.234567e6));
  test(SV("12_345_67_0"), loc, SV("{:L}"), F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("1_23_456_70_0"), loc, SV("{:L}"), F(1.234567e8));
    test(SV("1_2_34_567_00_0"), loc, SV("{:L}"), F(1.234567e9));
    test(SV("1_2_3_45_670_00_0"), loc, SV("{:L}"), F(1.234567e10));
    test(SV("1_2_3_4_56_700_00_0"), loc, SV("{:L}"), F(1.234567e11));
    test(SV("1#234567e+12"), loc, SV("{:L}"), F(1.234567e12));
    test(SV("1#234567e+13"), loc, SV("{:L}"), F(1.234567e13));
  }
  test(SV("-1#234567e-06"), loc, SV("{:L}"), F(-1.234567e-6));
  test(SV("-1#234567e-05"), loc, SV("{:L}"), F(-1.234567e-5));
  test(SV("-0#0001234567"), loc, SV("{:L}"), F(-1.234567e-4));
  test(SV("-0#001234567"), loc, SV("{:L}"), F(-1.234567e-3));
  test(SV("-0#01234567"), loc, SV("{:L}"), F(-1.234567e-2));
  test(SV("-0#1234567"), loc, SV("{:L}"), F(-1.234567e-1));
  test(SV("-1#234567"), loc, SV("{:L}"), F(-1.234567e0));
  test(SV("-1_2#34567"), loc, SV("{:L}"), F(-1.234567e1));
  test(SV("-12_3#4567"), loc, SV("{:L}"), F(-1.234567e2));
  test(SV("-1_23_4#567"), loc, SV("{:L}"), F(-1.234567e3));
  test(SV("-12_34_5#67"), loc, SV("{:L}"), F(-1.234567e4));
  test(SV("-123_45_6#7"), loc, SV("{:L}"), F(-1.234567e5));
  test(SV("-1_234_56_7"), loc, SV("{:L}"), F(-1.234567e6));
  test(SV("-12_345_67_0"), loc, SV("{:L}"), F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(SV("-1_23_456_70_0"), loc, SV("{:L}"), F(-1.234567e8));
    test(SV("-1_2_34_567_00_0"), loc, SV("{:L}"), F(-1.234567e9));
    test(SV("-1_2_3_45_670_00_0"), loc, SV("{:L}"), F(-1.234567e10));
    test(SV("-1_2_3_4_56_700_00_0"), loc, SV("{:L}"), F(-1.234567e11));
    test(SV("-1#234567e+12"), loc, SV("{:L}"), F(-1.234567e12));
    test(SV("-1#234567e+13"), loc, SV("{:L}"), F(-1.234567e13));
  }

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(SV("1,234.567$$$"), SV("{:$<12L}"), F(1.234567e3));
  test(SV("$$$1,234.567"), SV("{:$>12L}"), F(1.234567e3));
  test(SV("$1,234.567$$"), SV("{:$^12L}"), F(1.234567e3));
  test(SV("0001,234.567"), SV("{:012L}"), F(1.234567e3));
  test(SV("-1,234.567$$$"), SV("{:$<13L}"), F(-1.234567e3));
  test(SV("$$$-1,234.567"), SV("{:$>13L}"), F(-1.234567e3));
  test(SV("$-1,234.567$$"), SV("{:$^13L}"), F(-1.234567e3));
  test(SV("-0001,234.567"), SV("{:013L}"), F(-1.234567e3));

  std::locale::global(loc);
  test(SV("1_23_4#567$$$"), SV("{:$<13L}"), F(1.234567e3));
  test(SV("$$$1_23_4#567"), SV("{:$>13L}"), F(1.234567e3));
  test(SV("$1_23_4#567$$"), SV("{:$^13L}"), F(1.234567e3));
  test(SV("0001_23_4#567"), SV("{:013L}"), F(1.234567e3));
  test(SV("-1_23_4#567$$$"), SV("{:$<14L}"), F(-1.234567e3));
  test(SV("$$$-1_23_4#567"), SV("{:$>14L}"), F(-1.234567e3));
  test(SV("$-1_23_4#567$$"), SV("{:$^14L}"), F(-1.234567e3));
  test(SV("-0001_23_4#567"), SV("{:014L}"), F(-1.234567e3));

  test(SV("1,234.567$$$"), en_US, SV("{:$<12L}"), F(1.234567e3));
  test(SV("$$$1,234.567"), en_US, SV("{:$>12L}"), F(1.234567e3));
  test(SV("$1,234.567$$"), en_US, SV("{:$^12L}"), F(1.234567e3));
  test(SV("0001,234.567"), en_US, SV("{:012L}"), F(1.234567e3));
  test(SV("-1,234.567$$$"), en_US, SV("{:$<13L}"), F(-1.234567e3));
  test(SV("$$$-1,234.567"), en_US, SV("{:$>13L}"), F(-1.234567e3));
  test(SV("$-1,234.567$$"), en_US, SV("{:$^13L}"), F(-1.234567e3));
  test(SV("-0001,234.567"), en_US, SV("{:013L}"), F(-1.234567e3));

  std::locale::global(en_US);
  test(SV("1_23_4#567$$$"), loc, SV("{:$<13L}"), F(1.234567e3));
  test(SV("$$$1_23_4#567"), loc, SV("{:$>13L}"), F(1.234567e3));
  test(SV("$1_23_4#567$$"), loc, SV("{:$^13L}"), F(1.234567e3));
  test(SV("0001_23_4#567"), loc, SV("{:013L}"), F(1.234567e3));
  test(SV("-1_23_4#567$$$"), loc, SV("{:$<14L}"), F(-1.234567e3));
  test(SV("$$$-1_23_4#567"), loc, SV("{:$>14L}"), F(-1.234567e3));
  test(SV("$-1_23_4#567$$"), loc, SV("{:$^14L}"), F(-1.234567e3));
  test(SV("-0001_23_4#567"), loc, SV("{:014L}"), F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_default_precision() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(SV("1.23457e-06"), SV("{:.6L}"), F(1.234567e-6));
  test(SV("1.23457e-05"), SV("{:.6L}"), F(1.234567e-5));
  test(SV("0.000123457"), SV("{:.6L}"), F(1.234567e-4));
  test(SV("0.00123457"), SV("{:.6L}"), F(1.234567e-3));
  test(SV("0.0123457"), SV("{:.6L}"), F(1.234567e-2));
  test(SV("0.123457"), SV("{:.6L}"), F(1.234567e-1));
  test(SV("1.23457"), SV("{:.6L}"), F(1.234567e0));
  test(SV("12.3457"), SV("{:.6L}"), F(1.234567e1));
  test(SV("123.457"), SV("{:.6L}"), F(1.234567e2));
  test(SV("1,234.57"), SV("{:.6L}"), F(1.234567e3));
  test(SV("12,345.7"), SV("{:.6L}"), F(1.234567e4));
  test(SV("123,457"), SV("{:.6L}"), F(1.234567e5));
  test(SV("1.23457e+06"), SV("{:.6L}"), F(1.234567e6));
  test(SV("1.23457e+07"), SV("{:.6L}"), F(1.234567e7));
  test(SV("-1.23457e-06"), SV("{:.6L}"), F(-1.234567e-6));
  test(SV("-1.23457e-05"), SV("{:.6L}"), F(-1.234567e-5));
  test(SV("-0.000123457"), SV("{:.6L}"), F(-1.234567e-4));
  test(SV("-0.00123457"), SV("{:.6L}"), F(-1.234567e-3));
  test(SV("-0.0123457"), SV("{:.6L}"), F(-1.234567e-2));
  test(SV("-0.123457"), SV("{:.6L}"), F(-1.234567e-1));
  test(SV("-1.23457"), SV("{:.6L}"), F(-1.234567e0));
  test(SV("-12.3457"), SV("{:.6L}"), F(-1.234567e1));
  test(SV("-123.457"), SV("{:.6L}"), F(-1.234567e2));
  test(SV("-1,234.57"), SV("{:.6L}"), F(-1.234567e3));
  test(SV("-12,345.7"), SV("{:.6L}"), F(-1.234567e4));
  test(SV("-123,457"), SV("{:.6L}"), F(-1.234567e5));
  test(SV("-1.23457e+06"), SV("{:.6L}"), F(-1.234567e6));
  test(SV("-1.23457e+07"), SV("{:.6L}"), F(-1.234567e7));

  std::locale::global(loc);
  test(SV("1#23457e-06"), SV("{:.6L}"), F(1.234567e-6));
  test(SV("1#23457e-05"), SV("{:.6L}"), F(1.234567e-5));
  test(SV("0#000123457"), SV("{:.6L}"), F(1.234567e-4));
  test(SV("0#00123457"), SV("{:.6L}"), F(1.234567e-3));
  test(SV("0#0123457"), SV("{:.6L}"), F(1.234567e-2));
  test(SV("0#123457"), SV("{:.6L}"), F(1.234567e-1));
  test(SV("1#23457"), SV("{:.6L}"), F(1.234567e0));
  test(SV("1_2#3457"), SV("{:.6L}"), F(1.234567e1));
  test(SV("12_3#457"), SV("{:.6L}"), F(1.234567e2));
  test(SV("1_23_4#57"), SV("{:.6L}"), F(1.234567e3));
  test(SV("12_34_5#7"), SV("{:.6L}"), F(1.234567e4));
  test(SV("123_45_7"), SV("{:.6L}"), F(1.234567e5));
  test(SV("1#23457e+06"), SV("{:.6L}"), F(1.234567e6));
  test(SV("1#23457e+07"), SV("{:.6L}"), F(1.234567e7));
  test(SV("-1#23457e-06"), SV("{:.6L}"), F(-1.234567e-6));
  test(SV("-1#23457e-05"), SV("{:.6L}"), F(-1.234567e-5));
  test(SV("-0#000123457"), SV("{:.6L}"), F(-1.234567e-4));
  test(SV("-0#00123457"), SV("{:.6L}"), F(-1.234567e-3));
  test(SV("-0#0123457"), SV("{:.6L}"), F(-1.234567e-2));
  test(SV("-0#123457"), SV("{:.6L}"), F(-1.234567e-1));
  test(SV("-1#23457"), SV("{:.6L}"), F(-1.234567e0));
  test(SV("-1_2#3457"), SV("{:.6L}"), F(-1.234567e1));
  test(SV("-12_3#457"), SV("{:.6L}"), F(-1.234567e2));
  test(SV("-1_23_4#57"), SV("{:.6L}"), F(-1.234567e3));
  test(SV("-12_34_5#7"), SV("{:.6L}"), F(-1.234567e4));
  test(SV("-123_45_7"), SV("{:.6L}"), F(-1.234567e5));
  test(SV("-1#23457e+06"), SV("{:.6L}"), F(-1.234567e6));
  test(SV("-1#23457e+07"), SV("{:.6L}"), F(-1.234567e7));

  test(SV("1.23457e-06"), en_US, SV("{:.6L}"), F(1.234567e-6));
  test(SV("1.23457e-05"), en_US, SV("{:.6L}"), F(1.234567e-5));
  test(SV("0.000123457"), en_US, SV("{:.6L}"), F(1.234567e-4));
  test(SV("0.00123457"), en_US, SV("{:.6L}"), F(1.234567e-3));
  test(SV("0.0123457"), en_US, SV("{:.6L}"), F(1.234567e-2));
  test(SV("0.123457"), en_US, SV("{:.6L}"), F(1.234567e-1));
  test(SV("1.23457"), en_US, SV("{:.6L}"), F(1.234567e0));
  test(SV("12.3457"), en_US, SV("{:.6L}"), F(1.234567e1));
  test(SV("123.457"), en_US, SV("{:.6L}"), F(1.234567e2));
  test(SV("1,234.57"), en_US, SV("{:.6L}"), F(1.234567e3));
  test(SV("12,345.7"), en_US, SV("{:.6L}"), F(1.234567e4));
  test(SV("123,457"), en_US, SV("{:.6L}"), F(1.234567e5));
  test(SV("1.23457e+06"), en_US, SV("{:.6L}"), F(1.234567e6));
  test(SV("1.23457e+07"), en_US, SV("{:.6L}"), F(1.234567e7));
  test(SV("-1.23457e-06"), en_US, SV("{:.6L}"), F(-1.234567e-6));
  test(SV("-1.23457e-05"), en_US, SV("{:.6L}"), F(-1.234567e-5));
  test(SV("-0.000123457"), en_US, SV("{:.6L}"), F(-1.234567e-4));
  test(SV("-0.00123457"), en_US, SV("{:.6L}"), F(-1.234567e-3));
  test(SV("-0.0123457"), en_US, SV("{:.6L}"), F(-1.234567e-2));
  test(SV("-0.123457"), en_US, SV("{:.6L}"), F(-1.234567e-1));
  test(SV("-1.23457"), en_US, SV("{:.6L}"), F(-1.234567e0));
  test(SV("-12.3457"), en_US, SV("{:.6L}"), F(-1.234567e1));
  test(SV("-123.457"), en_US, SV("{:.6L}"), F(-1.234567e2));
  test(SV("-1,234.57"), en_US, SV("{:.6L}"), F(-1.234567e3));
  test(SV("-12,345.7"), en_US, SV("{:.6L}"), F(-1.234567e4));
  test(SV("-123,457"), en_US, SV("{:.6L}"), F(-1.234567e5));
  test(SV("-1.23457e+06"), en_US, SV("{:.6L}"), F(-1.234567e6));
  test(SV("-1.23457e+07"), en_US, SV("{:.6L}"), F(-1.234567e7));

  std::locale::global(en_US);
  test(SV("1#23457e-06"), loc, SV("{:.6L}"), F(1.234567e-6));
  test(SV("1#23457e-05"), loc, SV("{:.6L}"), F(1.234567e-5));
  test(SV("0#000123457"), loc, SV("{:.6L}"), F(1.234567e-4));
  test(SV("0#00123457"), loc, SV("{:.6L}"), F(1.234567e-3));
  test(SV("0#0123457"), loc, SV("{:.6L}"), F(1.234567e-2));
  test(SV("0#123457"), loc, SV("{:.6L}"), F(1.234567e-1));
  test(SV("1#23457"), loc, SV("{:.6L}"), F(1.234567e0));
  test(SV("1_2#3457"), loc, SV("{:.6L}"), F(1.234567e1));
  test(SV("12_3#457"), loc, SV("{:.6L}"), F(1.234567e2));
  test(SV("1_23_4#57"), loc, SV("{:.6L}"), F(1.234567e3));
  test(SV("12_34_5#7"), loc, SV("{:.6L}"), F(1.234567e4));
  test(SV("123_45_7"), loc, SV("{:.6L}"), F(1.234567e5));
  test(SV("1#23457e+06"), loc, SV("{:.6L}"), F(1.234567e6));
  test(SV("1#23457e+07"), loc, SV("{:.6L}"), F(1.234567e7));
  test(SV("-1#23457e-06"), loc, SV("{:.6L}"), F(-1.234567e-6));
  test(SV("-1#23457e-05"), loc, SV("{:.6L}"), F(-1.234567e-5));
  test(SV("-0#000123457"), loc, SV("{:.6L}"), F(-1.234567e-4));
  test(SV("-0#00123457"), loc, SV("{:.6L}"), F(-1.234567e-3));
  test(SV("-0#0123457"), loc, SV("{:.6L}"), F(-1.234567e-2));
  test(SV("-0#123457"), loc, SV("{:.6L}"), F(-1.234567e-1));
  test(SV("-1#23457"), loc, SV("{:.6L}"), F(-1.234567e0));
  test(SV("-1_2#3457"), loc, SV("{:.6L}"), F(-1.234567e1));
  test(SV("-12_3#457"), loc, SV("{:.6L}"), F(-1.234567e2));
  test(SV("-1_23_4#57"), loc, SV("{:.6L}"), F(-1.234567e3));
  test(SV("-12_34_5#7"), loc, SV("{:.6L}"), F(-1.234567e4));
  test(SV("-123_45_7"), loc, SV("{:.6L}"), F(-1.234567e5));
  test(SV("-1#23457e+06"), loc, SV("{:.6L}"), F(-1.234567e6));
  test(SV("-1#23457e+07"), loc, SV("{:.6L}"), F(-1.234567e7));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(SV("1,234.57$$$"), SV("{:$<11.6L}"), F(1.234567e3));
  test(SV("$$$1,234.57"), SV("{:$>11.6L}"), F(1.234567e3));
  test(SV("$1,234.57$$"), SV("{:$^11.6L}"), F(1.234567e3));
  test(SV("0001,234.57"), SV("{:011.6L}"), F(1.234567e3));
  test(SV("-1,234.57$$$"), SV("{:$<12.6L}"), F(-1.234567e3));
  test(SV("$$$-1,234.57"), SV("{:$>12.6L}"), F(-1.234567e3));
  test(SV("$-1,234.57$$"), SV("{:$^12.6L}"), F(-1.234567e3));
  test(SV("-0001,234.57"), SV("{:012.6L}"), F(-1.234567e3));

  std::locale::global(loc);
  test(SV("1_23_4#57$$$"), SV("{:$<12.6L}"), F(1.234567e3));
  test(SV("$$$1_23_4#57"), SV("{:$>12.6L}"), F(1.234567e3));
  test(SV("$1_23_4#57$$"), SV("{:$^12.6L}"), F(1.234567e3));
  test(SV("0001_23_4#57"), SV("{:012.6L}"), F(1.234567e3));
  test(SV("-1_23_4#57$$$"), SV("{:$<13.6L}"), F(-1.234567e3));
  test(SV("$$$-1_23_4#57"), SV("{:$>13.6L}"), F(-1.234567e3));
  test(SV("$-1_23_4#57$$"), SV("{:$^13.6L}"), F(-1.234567e3));
  test(SV("-0001_23_4#57"), SV("{:013.6L}"), F(-1.234567e3));

  test(SV("1,234.57$$$"), en_US, SV("{:$<11.6L}"), F(1.234567e3));
  test(SV("$$$1,234.57"), en_US, SV("{:$>11.6L}"), F(1.234567e3));
  test(SV("$1,234.57$$"), en_US, SV("{:$^11.6L}"), F(1.234567e3));
  test(SV("0001,234.57"), en_US, SV("{:011.6L}"), F(1.234567e3));
  test(SV("-1,234.57$$$"), en_US, SV("{:$<12.6L}"), F(-1.234567e3));
  test(SV("$$$-1,234.57"), en_US, SV("{:$>12.6L}"), F(-1.234567e3));
  test(SV("$-1,234.57$$"), en_US, SV("{:$^12.6L}"), F(-1.234567e3));
  test(SV("-0001,234.57"), en_US, SV("{:012.6L}"), F(-1.234567e3));

  std::locale::global(en_US);
  test(SV("1_23_4#57$$$"), loc, SV("{:$<12.6L}"), F(1.234567e3));
  test(SV("$$$1_23_4#57"), loc, SV("{:$>12.6L}"), F(1.234567e3));
  test(SV("$1_23_4#57$$"), loc, SV("{:$^12.6L}"), F(1.234567e3));
  test(SV("0001_23_4#57"), loc, SV("{:012.6L}"), F(1.234567e3));
  test(SV("-1_23_4#57$$$"), loc, SV("{:$<13.6L}"), F(-1.234567e3));
  test(SV("$$$-1_23_4#57"), loc, SV("{:$>13.6L}"), F(-1.234567e3));
  test(SV("$-1_23_4#57$$"), loc, SV("{:$^13.6L}"), F(-1.234567e3));
  test(SV("-0001_23_4#57"), loc, SV("{:013.6L}"), F(-1.234567e3));
}

template <class F, class CharT >
void test_floating_point() {
  test_floating_point_hex_lower_case<F, CharT>();
  test_floating_point_hex_upper_case<F, CharT>();
  test_floating_point_hex_lower_case_precision<F, CharT>();
  test_floating_point_hex_upper_case_precision<F, CharT>();

  test_floating_point_scientific_lower_case<F, CharT>();
  test_floating_point_scientific_upper_case<F, CharT>();

  test_floating_point_fixed_lower_case<F, CharT>();
  test_floating_point_fixed_upper_case<F, CharT>();

  test_floating_point_general_lower_case<F, CharT>();
  test_floating_point_general_upper_case<F, CharT>();

  test_floating_point_default<F, CharT>();
  test_floating_point_default_precision<F, CharT>();
}

template <class CharT>
void test() {
  test_bool<CharT>();
  test_integer<CharT>();
  test_floating_point<float, CharT>();
  test_floating_point<double, CharT>();
  test_floating_point<long double, CharT>();
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
