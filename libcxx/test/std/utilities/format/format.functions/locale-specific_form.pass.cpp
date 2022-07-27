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

template <string_literal fmt, class CharT, class... Args>
void test(std::basic_string_view<CharT> expected, const Args&... args) {
  // *** format ***
  {
    std::basic_string<CharT> out = std::format(fmt.template sv<CharT>(), args...);
    if constexpr (std::same_as<CharT, char>)
      if (out != expected)
        std::cerr << "\nFormat string   " << fmt.template sv<char>() << "\nExpected output " << expected
                  << "\nActual output   " << out << '\n';
    assert(out == expected);
  }
  // *** vformat ***
  {
    std::basic_string<CharT> out =
        std::vformat(fmt.template sv<CharT>(), std::make_format_args<context_t<CharT>>(args...));
    assert(out == expected);
  }
  // *** format_to ***
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::format_to(out.begin(), fmt.template sv<CharT>(), args...);
    assert(it == out.end());
    assert(out == expected);
  }
  // *** vformat_to ***
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::vformat_to(out.begin(), fmt.template sv<CharT>(), std::make_format_args<context_t<CharT>>(args...));
    assert(it == out.end());
    assert(out == expected);
  }
  // *** format_to_n ***
  {
    std::basic_string<CharT> out;
    std::format_to_n_result result = std::format_to_n(std::back_inserter(out), 1000, fmt.template sv<CharT>(), args...);
    using diff_type = decltype(result.size);
    diff_type formatted_size = std::formatted_size(fmt.template sv<CharT>(), args...);
    diff_type size = std::min<diff_type>(1000, formatted_size);

    assert(result.size == formatted_size);
    assert(out == expected.substr(0, size));
  }
  // *** formatted_size ***
  {
    size_t size = std::formatted_size(fmt.template sv<CharT>(), args...);
    assert(size == expected.size());
  }
}

template <string_literal fmt, class CharT, class... Args>
void test(std::basic_string_view<CharT> expected, std::locale loc, const Args&... args) {
  // *** format ***
  {
    std::basic_string<CharT> out = std::format(loc, fmt.template sv<CharT>(), args...);
    if constexpr (std::same_as<CharT, char>)
      if (out != expected)
        std::cerr << "\nFormat string   " << fmt.template sv<char>() << "\nExpected output " << expected
                  << "\nActual output   " << out << '\n';
    assert(out == expected);
  }
  // *** vformat ***
  {
    std::basic_string<CharT> out =
        std::vformat(loc, fmt.template sv<CharT>(), std::make_format_args<context_t<CharT>>(args...));
    assert(out == expected);
  }
  // *** format_to ***
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::format_to(out.begin(), loc, fmt.template sv<CharT>(), args...);
    assert(it == out.end());
    assert(out == expected);
  }
  // *** vformat_to ***
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it =
        std::vformat_to(out.begin(), loc, fmt.template sv<CharT>(), std::make_format_args<context_t<CharT>>(args...));
    assert(it == out.end());
    assert(out == expected);
  }
  // *** format_to_n ***
  {
    std::basic_string<CharT> out;
    std::format_to_n_result result =
        std::format_to_n(std::back_inserter(out), 1000, loc, fmt.template sv<CharT>(), args...);
    using diff_type = decltype(result.size);
    diff_type formatted_size = std::formatted_size(loc, fmt.template sv<CharT>(), args...);
    diff_type size = std::min<diff_type>(1000, formatted_size);

    assert(result.size == formatted_size);
    assert(out == expected.substr(0, size));
  }
  // *** formatted_size ***
  {
    size_t size = std::formatted_size(loc, fmt.template sv<CharT>(), args...);
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
  test<"{:L}">(SV("true"), true);
  test<"{:L}">(SV("false"), false);

  test<"{:L}">(SV("yes"), loc, true);
  test<"{:L}">(SV("no"), loc, false);

  std::locale::global(loc);
  test<"{:L}">(SV("yes"), true);
  test<"{:L}">(SV("no"), false);

  test<"{:L}">(SV("true"), std::locale(LOCALE_en_US_UTF_8), true);
  test<"{:L}">(SV("false"), std::locale(LOCALE_en_US_UTF_8), false);

#ifndef TEST_HAS_NO_UNICODE
  std::locale loc_unicode = std::locale(std::locale(), new numpunct_unicode<CharT>());

  test<"{:L}">(SV("gültig"), loc_unicode, true);
  test<"{:L}">(SV("ungültig"), loc_unicode, false);

  test<"{:9L}">(SV("gültig   "), loc_unicode, true);
  test<"{:!<9L}">(SV("gültig!!!"), loc_unicode, true);
  test<"{:_^9L}">(SV("_gültig__"), loc_unicode, true);
  test<"{:>9L}">(SV("   gültig"), loc_unicode, true);
#endif // TEST_HAS_NO_UNICODE
}

template <class CharT>
void test_integer() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Decimal ***
  std::locale::global(en_US);
  test<"{:L}">(SV("0"), 0);
  test<"{:L}">(SV("1"), 1);
  test<"{:L}">(SV("10"), 10);
  test<"{:L}">(SV("100"), 100);
  test<"{:L}">(SV("1,000"), 1'000);
  test<"{:L}">(SV("10,000"), 10'000);
  test<"{:L}">(SV("100,000"), 100'000);
  test<"{:L}">(SV("1,000,000"), 1'000'000);
  test<"{:L}">(SV("10,000,000"), 10'000'000);
  test<"{:L}">(SV("100,000,000"), 100'000'000);
  test<"{:L}">(SV("1,000,000,000"), 1'000'000'000);

  test<"{:L}">(SV("-1"), -1);
  test<"{:L}">(SV("-10"), -10);
  test<"{:L}">(SV("-100"), -100);
  test<"{:L}">(SV("-1,000"), -1'000);
  test<"{:L}">(SV("-10,000"), -10'000);
  test<"{:L}">(SV("-100,000"), -100'000);
  test<"{:L}">(SV("-1,000,000"), -1'000'000);
  test<"{:L}">(SV("-10,000,000"), -10'000'000);
  test<"{:L}">(SV("-100,000,000"), -100'000'000);
  test<"{:L}">(SV("-1,000,000,000"), -1'000'000'000);

  std::locale::global(loc);
  test<"{:L}">(SV("0"), 0);
  test<"{:L}">(SV("1"), 1);
  test<"{:L}">(SV("1_0"), 10);
  test<"{:L}">(SV("10_0"), 100);
  test<"{:L}">(SV("1_00_0"), 1'000);
  test<"{:L}">(SV("10_00_0"), 10'000);
  test<"{:L}">(SV("100_00_0"), 100'000);
  test<"{:L}">(SV("1_000_00_0"), 1'000'000);
  test<"{:L}">(SV("10_000_00_0"), 10'000'000);
  test<"{:L}">(SV("1_00_000_00_0"), 100'000'000);
  test<"{:L}">(SV("1_0_00_000_00_0"), 1'000'000'000);

  test<"{:L}">(SV("-1"), -1);
  test<"{:L}">(SV("-1_0"), -10);
  test<"{:L}">(SV("-10_0"), -100);
  test<"{:L}">(SV("-1_00_0"), -1'000);
  test<"{:L}">(SV("-10_00_0"), -10'000);
  test<"{:L}">(SV("-100_00_0"), -100'000);
  test<"{:L}">(SV("-1_000_00_0"), -1'000'000);
  test<"{:L}">(SV("-10_000_00_0"), -10'000'000);
  test<"{:L}">(SV("-1_00_000_00_0"), -100'000'000);
  test<"{:L}">(SV("-1_0_00_000_00_0"), -1'000'000'000);

  test<"{:L}">(SV("0"), en_US, 0);
  test<"{:L}">(SV("1"), en_US, 1);
  test<"{:L}">(SV("10"), en_US, 10);
  test<"{:L}">(SV("100"), en_US, 100);
  test<"{:L}">(SV("1,000"), en_US, 1'000);
  test<"{:L}">(SV("10,000"), en_US, 10'000);
  test<"{:L}">(SV("100,000"), en_US, 100'000);
  test<"{:L}">(SV("1,000,000"), en_US, 1'000'000);
  test<"{:L}">(SV("10,000,000"), en_US, 10'000'000);
  test<"{:L}">(SV("100,000,000"), en_US, 100'000'000);
  test<"{:L}">(SV("1,000,000,000"), en_US, 1'000'000'000);

  test<"{:L}">(SV("-1"), en_US, -1);
  test<"{:L}">(SV("-10"), en_US, -10);
  test<"{:L}">(SV("-100"), en_US, -100);
  test<"{:L}">(SV("-1,000"), en_US, -1'000);
  test<"{:L}">(SV("-10,000"), en_US, -10'000);
  test<"{:L}">(SV("-100,000"), en_US, -100'000);
  test<"{:L}">(SV("-1,000,000"), en_US, -1'000'000);
  test<"{:L}">(SV("-10,000,000"), en_US, -10'000'000);
  test<"{:L}">(SV("-100,000,000"), en_US, -100'000'000);
  test<"{:L}">(SV("-1,000,000,000"), en_US, -1'000'000'000);

  std::locale::global(en_US);
  test<"{:L}">(SV("0"), loc, 0);
  test<"{:L}">(SV("1"), loc, 1);
  test<"{:L}">(SV("1_0"), loc, 10);
  test<"{:L}">(SV("10_0"), loc, 100);
  test<"{:L}">(SV("1_00_0"), loc, 1'000);
  test<"{:L}">(SV("10_00_0"), loc, 10'000);
  test<"{:L}">(SV("100_00_0"), loc, 100'000);
  test<"{:L}">(SV("1_000_00_0"), loc, 1'000'000);
  test<"{:L}">(SV("10_000_00_0"), loc, 10'000'000);
  test<"{:L}">(SV("1_00_000_00_0"), loc, 100'000'000);
  test<"{:L}">(SV("1_0_00_000_00_0"), loc, 1'000'000'000);

  test<"{:L}">(SV("-1"), loc, -1);
  test<"{:L}">(SV("-1_0"), loc, -10);
  test<"{:L}">(SV("-10_0"), loc, -100);
  test<"{:L}">(SV("-1_00_0"), loc, -1'000);
  test<"{:L}">(SV("-10_00_0"), loc, -10'000);
  test<"{:L}">(SV("-100_00_0"), loc, -100'000);
  test<"{:L}">(SV("-1_000_00_0"), loc, -1'000'000);
  test<"{:L}">(SV("-10_000_00_0"), loc, -10'000'000);
  test<"{:L}">(SV("-1_00_000_00_0"), loc, -100'000'000);
  test<"{:L}">(SV("-1_0_00_000_00_0"), loc, -1'000'000'000);

  // *** Binary ***
  std::locale::global(en_US);
  test<"{:Lb}">(SV("0"), 0b0);
  test<"{:Lb}">(SV("1"), 0b1);
  test<"{:Lb}">(SV("1,000,000,000"), 0b1'000'000'000);

  test<"{:#Lb}">(SV("0b0"), 0b0);
  test<"{:#Lb}">(SV("0b1"), 0b1);
  test<"{:#Lb}">(SV("0b1,000,000,000"), 0b1'000'000'000);

  test<"{:LB}">(SV("-1"), -0b1);
  test<"{:LB}">(SV("-1,000,000,000"), -0b1'000'000'000);

  test<"{:#LB}">(SV("-0B1"), -0b1);
  test<"{:#LB}">(SV("-0B1,000,000,000"), -0b1'000'000'000);

  std::locale::global(loc);
  test<"{:Lb}">(SV("0"), 0b0);
  test<"{:Lb}">(SV("1"), 0b1);
  test<"{:Lb}">(SV("1_0_00_000_00_0"), 0b1'000'000'000);

  test<"{:#Lb}">(SV("0b0"), 0b0);
  test<"{:#Lb}">(SV("0b1"), 0b1);
  test<"{:#Lb}">(SV("0b1_0_00_000_00_0"), 0b1'000'000'000);

  test<"{:LB}">(SV("-1"), -0b1);
  test<"{:LB}">(SV("-1_0_00_000_00_0"), -0b1'000'000'000);

  test<"{:#LB}">(SV("-0B1"), -0b1);
  test<"{:#LB}">(SV("-0B1_0_00_000_00_0"), -0b1'000'000'000);

  test<"{:Lb}">(SV("0"), en_US, 0b0);
  test<"{:Lb}">(SV("1"), en_US, 0b1);
  test<"{:Lb}">(SV("1,000,000,000"), en_US, 0b1'000'000'000);

  test<"{:#Lb}">(SV("0b0"), en_US, 0b0);
  test<"{:#Lb}">(SV("0b1"), en_US, 0b1);
  test<"{:#Lb}">(SV("0b1,000,000,000"), en_US, 0b1'000'000'000);

  test<"{:LB}">(SV("-1"), en_US, -0b1);
  test<"{:LB}">(SV("-1,000,000,000"), en_US, -0b1'000'000'000);

  test<"{:#LB}">(SV("-0B1"), en_US, -0b1);
  test<"{:#LB}">(SV("-0B1,000,000,000"), en_US, -0b1'000'000'000);

  std::locale::global(en_US);
  test<"{:Lb}">(SV("0"), loc, 0b0);
  test<"{:Lb}">(SV("1"), loc, 0b1);
  test<"{:Lb}">(SV("1_0_00_000_00_0"), loc, 0b1'000'000'000);

  test<"{:#Lb}">(SV("0b0"), loc, 0b0);
  test<"{:#Lb}">(SV("0b1"), loc, 0b1);
  test<"{:#Lb}">(SV("0b1_0_00_000_00_0"), loc, 0b1'000'000'000);

  test<"{:LB}">(SV("-1"), loc, -0b1);
  test<"{:LB}">(SV("-1_0_00_000_00_0"), loc, -0b1'000'000'000);

  test<"{:#LB}">(SV("-0B1"), loc, -0b1);
  test<"{:#LB}">(SV("-0B1_0_00_000_00_0"), loc, -0b1'000'000'000);

  // *** Octal ***
  std::locale::global(en_US);
  test<"{:Lo}">(SV("0"), 00);
  test<"{:Lo}">(SV("1"), 01);
  test<"{:Lo}">(SV("1,000,000,000"), 01'000'000'000);

  test<"{:#Lo}">(SV("0"), 00);
  test<"{:#Lo}">(SV("01"), 01);
  test<"{:#Lo}">(SV("01,000,000,000"), 01'000'000'000);

  test<"{:Lo}">(SV("-1"), -01);
  test<"{:Lo}">(SV("-1,000,000,000"), -01'000'000'000);

  test<"{:#Lo}">(SV("-01"), -01);
  test<"{:#Lo}">(SV("-01,000,000,000"), -01'000'000'000);

  std::locale::global(loc);
  test<"{:Lo}">(SV("0"), 00);
  test<"{:Lo}">(SV("1"), 01);
  test<"{:Lo}">(SV("1_0_00_000_00_0"), 01'000'000'000);

  test<"{:#Lo}">(SV("0"), 00);
  test<"{:#Lo}">(SV("01"), 01);
  test<"{:#Lo}">(SV("01_0_00_000_00_0"), 01'000'000'000);

  test<"{:Lo}">(SV("-1"), -01);
  test<"{:Lo}">(SV("-1_0_00_000_00_0"), -01'000'000'000);

  test<"{:#Lo}">(SV("-01"), -01);
  test<"{:#Lo}">(SV("-01_0_00_000_00_0"), -01'000'000'000);

  test<"{:Lo}">(SV("0"), en_US, 00);
  test<"{:Lo}">(SV("1"), en_US, 01);
  test<"{:Lo}">(SV("1,000,000,000"), en_US, 01'000'000'000);

  test<"{:#Lo}">(SV("0"), en_US, 00);
  test<"{:#Lo}">(SV("01"), en_US, 01);
  test<"{:#Lo}">(SV("01,000,000,000"), en_US, 01'000'000'000);

  test<"{:Lo}">(SV("-1"), en_US, -01);
  test<"{:Lo}">(SV("-1,000,000,000"), en_US, -01'000'000'000);

  test<"{:#Lo}">(SV("-01"), en_US, -01);
  test<"{:#Lo}">(SV("-01,000,000,000"), en_US, -01'000'000'000);

  std::locale::global(en_US);
  test<"{:Lo}">(SV("0"), loc, 00);
  test<"{:Lo}">(SV("1"), loc, 01);
  test<"{:Lo}">(SV("1_0_00_000_00_0"), loc, 01'000'000'000);

  test<"{:#Lo}">(SV("0"), loc, 00);
  test<"{:#Lo}">(SV("01"), loc, 01);
  test<"{:#Lo}">(SV("01_0_00_000_00_0"), loc, 01'000'000'000);

  test<"{:Lo}">(SV("-1"), loc, -01);
  test<"{:Lo}">(SV("-1_0_00_000_00_0"), loc, -01'000'000'000);

  test<"{:#Lo}">(SV("-01"), loc, -01);
  test<"{:#Lo}">(SV("-01_0_00_000_00_0"), loc, -01'000'000'000);

  // *** Hexadecimal ***
  std::locale::global(en_US);
  test<"{:Lx}">(SV("0"), 0x0);
  test<"{:Lx}">(SV("1"), 0x1);
  test<"{:Lx}">(SV("1,000,000,000"), 0x1'000'000'000);

  test<"{:#Lx}">(SV("0x0"), 0x0);
  test<"{:#Lx}">(SV("0x1"), 0x1);
  test<"{:#Lx}">(SV("0x1,000,000,000"), 0x1'000'000'000);

  test<"{:LX}">(SV("-1"), -0x1);
  test<"{:LX}">(SV("-1,000,000,000"), -0x1'000'000'000);

  test<"{:#LX}">(SV("-0X1"), -0x1);
  test<"{:#LX}">(SV("-0X1,000,000,000"), -0x1'000'000'000);

  std::locale::global(loc);
  test<"{:Lx}">(SV("0"), 0x0);
  test<"{:Lx}">(SV("1"), 0x1);
  test<"{:Lx}">(SV("1_0_00_000_00_0"), 0x1'000'000'000);

  test<"{:#Lx}">(SV("0x0"), 0x0);
  test<"{:#Lx}">(SV("0x1"), 0x1);
  test<"{:#Lx}">(SV("0x1_0_00_000_00_0"), 0x1'000'000'000);

  test<"{:LX}">(SV("-1"), -0x1);
  test<"{:LX}">(SV("-1_0_00_000_00_0"), -0x1'000'000'000);

  test<"{:#LX}">(SV("-0X1"), -0x1);
  test<"{:#LX}">(SV("-0X1_0_00_000_00_0"), -0x1'000'000'000);

  test<"{:Lx}">(SV("0"), en_US, 0x0);
  test<"{:Lx}">(SV("1"), en_US, 0x1);
  test<"{:Lx}">(SV("1,000,000,000"), en_US, 0x1'000'000'000);

  test<"{:#Lx}">(SV("0x0"), en_US, 0x0);
  test<"{:#Lx}">(SV("0x1"), en_US, 0x1);
  test<"{:#Lx}">(SV("0x1,000,000,000"), en_US, 0x1'000'000'000);

  test<"{:LX}">(SV("-1"), en_US, -0x1);
  test<"{:LX}">(SV("-1,000,000,000"), en_US, -0x1'000'000'000);

  test<"{:#LX}">(SV("-0X1"), en_US, -0x1);
  test<"{:#LX}">(SV("-0X1,000,000,000"), en_US, -0x1'000'000'000);

  std::locale::global(en_US);
  test<"{:Lx}">(SV("0"), loc, 0x0);
  test<"{:Lx}">(SV("1"), loc, 0x1);
  test<"{:Lx}">(SV("1_0_00_000_00_0"), loc, 0x1'000'000'000);

  test<"{:#Lx}">(SV("0x0"), loc, 0x0);
  test<"{:#Lx}">(SV("0x1"), loc, 0x1);
  test<"{:#Lx}">(SV("0x1_0_00_000_00_0"), loc, 0x1'000'000'000);

  test<"{:LX}">(SV("-1"), loc, -0x1);
  test<"{:LX}">(SV("-1_0_00_000_00_0"), loc, -0x1'000'000'000);

  test<"{:#LX}">(SV("-0X1"), loc, -0x1);
  test<"{:#LX}">(SV("-0X1_0_00_000_00_0"), loc, -0x1'000'000'000);

  // *** align-fill & width ***
  test<"{:L}">(SV("4_2"), loc, 42);

  test<"{:6L}">(SV("   4_2"), loc, 42);
  test<"{:<6L}">(SV("4_2   "), loc, 42);
  test<"{:^6L}">(SV(" 4_2  "), loc, 42);
  test<"{:>6L}">(SV("   4_2"), loc, 42);

  test<"{:*<6L}">(SV("4_2***"), loc, 42);
  test<"{:*^6L}">(SV("*4_2**"), loc, 42);
  test<"{:*>6L}">(SV("***4_2"), loc, 42);

  test<"{:*<8Lx}">(SV("4_a*****"), loc, 0x4a);
  test<"{:*^8Lx}">(SV("**4_a***"), loc, 0x4a);
  test<"{:*>8Lx}">(SV("*****4_a"), loc, 0x4a);

  test<"{:*<#8Lx}">(SV("0x4_a***"), loc, 0x4a);
  test<"{:*^#8Lx}">(SV("*0x4_a**"), loc, 0x4a);
  test<"{:*>#8Lx}">(SV("***0x4_a"), loc, 0x4a);

  test<"{:*<8LX}">(SV("4_A*****"), loc, 0x4a);
  test<"{:*^8LX}">(SV("**4_A***"), loc, 0x4a);
  test<"{:*>8LX}">(SV("*****4_A"), loc, 0x4a);

  test<"{:*<#8LX}">(SV("0X4_A***"), loc, 0x4a);
  test<"{:*^#8LX}">(SV("*0X4_A**"), loc, 0x4a);
  test<"{:*>#8LX}">(SV("***0X4_A"), loc, 0x4a);

  // Test whether zero padding is ignored
  test<"{:<06L}">(SV("4_2   "), loc, 42);
  test<"{:^06L}">(SV(" 4_2  "), loc, 42);
  test<"{:>06L}">(SV("   4_2"), loc, 42);

  // *** zero-padding & width ***
  test<"{:6L}">(SV("   4_2"), loc, 42);
  test<"{:06L}">(SV("0004_2"), loc, 42);
  test<"{:06L}">(SV("-004_2"), loc, -42);

  test<"{:08Lx}">(SV("000004_a"), loc, 0x4a);
  test<"{:#08Lx}">(SV("0x0004_a"), loc, 0x4a);
  test<"{:#08LX}">(SV("0X0004_A"), loc, 0x4a);

  test<"{:08Lx}">(SV("-00004_a"), loc, -0x4a);
  test<"{:#08Lx}">(SV("-0x004_a"), loc, -0x4a);
  test<"{:#08LX}">(SV("-0X004_A"), loc, -0x4a);
}

template <class F, class CharT>
void test_floating_point_hex_lower_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test<"{:La}">(SV("1.23456p-3"), F(0x1.23456p-3));
  test<"{:La}">(SV("1.23456p-2"), F(0x1.23456p-2));
  test<"{:La}">(SV("1.23456p-1"), F(0x1.23456p-1));
  test<"{:La}">(SV("1.23456p+0"), F(0x1.23456p0));
  test<"{:La}">(SV("1.23456p+1"), F(0x1.23456p+1));
  test<"{:La}">(SV("1.23456p+2"), F(0x1.23456p+2));
  test<"{:La}">(SV("1.23456p+3"), F(0x1.23456p+3));
  test<"{:La}">(SV("1.23456p+20"), F(0x1.23456p+20));

  std::locale::global(loc);
  test<"{:La}">(SV("1#23456p-3"), F(0x1.23456p-3));
  test<"{:La}">(SV("1#23456p-2"), F(0x1.23456p-2));
  test<"{:La}">(SV("1#23456p-1"), F(0x1.23456p-1));
  test<"{:La}">(SV("1#23456p+0"), F(0x1.23456p0));
  test<"{:La}">(SV("1#23456p+1"), F(0x1.23456p+1));
  test<"{:La}">(SV("1#23456p+2"), F(0x1.23456p+2));
  test<"{:La}">(SV("1#23456p+3"), F(0x1.23456p+3));
  test<"{:La}">(SV("1#23456p+20"), F(0x1.23456p+20));

  test<"{:La}">(SV("1.23456p-3"), en_US, F(0x1.23456p-3));
  test<"{:La}">(SV("1.23456p-2"), en_US, F(0x1.23456p-2));
  test<"{:La}">(SV("1.23456p-1"), en_US, F(0x1.23456p-1));
  test<"{:La}">(SV("1.23456p+0"), en_US, F(0x1.23456p0));
  test<"{:La}">(SV("1.23456p+1"), en_US, F(0x1.23456p+1));
  test<"{:La}">(SV("1.23456p+2"), en_US, F(0x1.23456p+2));
  test<"{:La}">(SV("1.23456p+3"), en_US, F(0x1.23456p+3));
  test<"{:La}">(SV("1.23456p+20"), en_US, F(0x1.23456p+20));

  std::locale::global(en_US);
  test<"{:La}">(SV("1#23456p-3"), loc, F(0x1.23456p-3));
  test<"{:La}">(SV("1#23456p-2"), loc, F(0x1.23456p-2));
  test<"{:La}">(SV("1#23456p-1"), loc, F(0x1.23456p-1));
  test<"{:La}">(SV("1#23456p+0"), loc, F(0x1.23456p0));
  test<"{:La}">(SV("1#23456p+1"), loc, F(0x1.23456p+1));
  test<"{:La}">(SV("1#23456p+2"), loc, F(0x1.23456p+2));
  test<"{:La}">(SV("1#23456p+3"), loc, F(0x1.23456p+3));
  test<"{:La}">(SV("1#23456p+20"), loc, F(0x1.23456p+20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test<"{:$<13La}">(SV("1.23456p+3$$$"), F(0x1.23456p3));
  test<"{:$>13La}">(SV("$$$1.23456p+3"), F(0x1.23456p3));
  test<"{:$^13La}">(SV("$1.23456p+3$$"), F(0x1.23456p3));
  test<"{:013La}">(SV("0001.23456p+3"), F(0x1.23456p3));
  test<"{:$<14La}">(SV("-1.23456p+3$$$"), F(-0x1.23456p3));
  test<"{:$>14La}">(SV("$$$-1.23456p+3"), F(-0x1.23456p3));
  test<"{:$^14La}">(SV("$-1.23456p+3$$"), F(-0x1.23456p3));
  test<"{:014La}">(SV("-0001.23456p+3"), F(-0x1.23456p3));

  std::locale::global(loc);
  test<"{:$<13La}">(SV("1#23456p+3$$$"), F(0x1.23456p3));
  test<"{:$>13La}">(SV("$$$1#23456p+3"), F(0x1.23456p3));
  test<"{:$^13La}">(SV("$1#23456p+3$$"), F(0x1.23456p3));
  test<"{:013La}">(SV("0001#23456p+3"), F(0x1.23456p3));
  test<"{:$<14La}">(SV("-1#23456p+3$$$"), F(-0x1.23456p3));
  test<"{:$>14La}">(SV("$$$-1#23456p+3"), F(-0x1.23456p3));
  test<"{:$^14La}">(SV("$-1#23456p+3$$"), F(-0x1.23456p3));
  test<"{:014La}">(SV("-0001#23456p+3"), F(-0x1.23456p3));

  test<"{:$<13La}">(SV("1.23456p+3$$$"), en_US, F(0x1.23456p3));
  test<"{:$>13La}">(SV("$$$1.23456p+3"), en_US, F(0x1.23456p3));
  test<"{:$^13La}">(SV("$1.23456p+3$$"), en_US, F(0x1.23456p3));
  test<"{:013La}">(SV("0001.23456p+3"), en_US, F(0x1.23456p3));
  test<"{:$<14La}">(SV("-1.23456p+3$$$"), en_US, F(-0x1.23456p3));
  test<"{:$>14La}">(SV("$$$-1.23456p+3"), en_US, F(-0x1.23456p3));
  test<"{:$^14La}">(SV("$-1.23456p+3$$"), en_US, F(-0x1.23456p3));
  test<"{:014La}">(SV("-0001.23456p+3"), en_US, F(-0x1.23456p3));

  std::locale::global(en_US);
  test<"{:$<13La}">(SV("1#23456p+3$$$"), loc, F(0x1.23456p3));
  test<"{:$>13La}">(SV("$$$1#23456p+3"), loc, F(0x1.23456p3));
  test<"{:$^13La}">(SV("$1#23456p+3$$"), loc, F(0x1.23456p3));
  test<"{:013La}">(SV("0001#23456p+3"), loc, F(0x1.23456p3));
  test<"{:$<14La}">(SV("-1#23456p+3$$$"), loc, F(-0x1.23456p3));
  test<"{:$>14La}">(SV("$$$-1#23456p+3"), loc, F(-0x1.23456p3));
  test<"{:$^14La}">(SV("$-1#23456p+3$$"), loc, F(-0x1.23456p3));
  test<"{:014La}">(SV("-0001#23456p+3"), loc, F(-0x1.23456p3));
}

template <class F, class CharT>
void test_floating_point_hex_upper_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test<"{:LA}">(SV("1.23456P-3"), F(0x1.23456p-3));
  test<"{:LA}">(SV("1.23456P-2"), F(0x1.23456p-2));
  test<"{:LA}">(SV("1.23456P-1"), F(0x1.23456p-1));
  test<"{:LA}">(SV("1.23456P+0"), F(0x1.23456p0));
  test<"{:LA}">(SV("1.23456P+1"), F(0x1.23456p+1));
  test<"{:LA}">(SV("1.23456P+2"), F(0x1.23456p+2));
  test<"{:LA}">(SV("1.23456P+3"), F(0x1.23456p+3));
  test<"{:LA}">(SV("1.23456P+20"), F(0x1.23456p+20));

  std::locale::global(loc);
  test<"{:LA}">(SV("1#23456P-3"), F(0x1.23456p-3));
  test<"{:LA}">(SV("1#23456P-2"), F(0x1.23456p-2));
  test<"{:LA}">(SV("1#23456P-1"), F(0x1.23456p-1));
  test<"{:LA}">(SV("1#23456P+0"), F(0x1.23456p0));
  test<"{:LA}">(SV("1#23456P+1"), F(0x1.23456p+1));
  test<"{:LA}">(SV("1#23456P+2"), F(0x1.23456p+2));
  test<"{:LA}">(SV("1#23456P+3"), F(0x1.23456p+3));
  test<"{:LA}">(SV("1#23456P+20"), F(0x1.23456p+20));

  test<"{:LA}">(SV("1.23456P-3"), en_US, F(0x1.23456p-3));
  test<"{:LA}">(SV("1.23456P-2"), en_US, F(0x1.23456p-2));
  test<"{:LA}">(SV("1.23456P-1"), en_US, F(0x1.23456p-1));
  test<"{:LA}">(SV("1.23456P+0"), en_US, F(0x1.23456p0));
  test<"{:LA}">(SV("1.23456P+1"), en_US, F(0x1.23456p+1));
  test<"{:LA}">(SV("1.23456P+2"), en_US, F(0x1.23456p+2));
  test<"{:LA}">(SV("1.23456P+3"), en_US, F(0x1.23456p+3));
  test<"{:LA}">(SV("1.23456P+20"), en_US, F(0x1.23456p+20));

  std::locale::global(en_US);
  test<"{:LA}">(SV("1#23456P-3"), loc, F(0x1.23456p-3));
  test<"{:LA}">(SV("1#23456P-2"), loc, F(0x1.23456p-2));
  test<"{:LA}">(SV("1#23456P-1"), loc, F(0x1.23456p-1));
  test<"{:LA}">(SV("1#23456P+0"), loc, F(0x1.23456p0));
  test<"{:LA}">(SV("1#23456P+1"), loc, F(0x1.23456p+1));
  test<"{:LA}">(SV("1#23456P+2"), loc, F(0x1.23456p+2));
  test<"{:LA}">(SV("1#23456P+3"), loc, F(0x1.23456p+3));
  test<"{:LA}">(SV("1#23456P+20"), loc, F(0x1.23456p+20));

  // *** Fill, align, zero Padding ***
  std::locale::global(en_US);
  test<"{:$<13LA}">(SV("1.23456P+3$$$"), F(0x1.23456p3));
  test<"{:$>13LA}">(SV("$$$1.23456P+3"), F(0x1.23456p3));
  test<"{:$^13LA}">(SV("$1.23456P+3$$"), F(0x1.23456p3));
  test<"{:013LA}">(SV("0001.23456P+3"), F(0x1.23456p3));
  test<"{:$<14LA}">(SV("-1.23456P+3$$$"), F(-0x1.23456p3));
  test<"{:$>14LA}">(SV("$$$-1.23456P+3"), F(-0x1.23456p3));
  test<"{:$^14LA}">(SV("$-1.23456P+3$$"), F(-0x1.23456p3));
  test<"{:014LA}">(SV("-0001.23456P+3"), F(-0x1.23456p3));

  std::locale::global(loc);
  test<"{:$<13LA}">(SV("1#23456P+3$$$"), F(0x1.23456p3));
  test<"{:$>13LA}">(SV("$$$1#23456P+3"), F(0x1.23456p3));
  test<"{:$^13LA}">(SV("$1#23456P+3$$"), F(0x1.23456p3));
  test<"{:013LA}">(SV("0001#23456P+3"), F(0x1.23456p3));
  test<"{:$<14LA}">(SV("-1#23456P+3$$$"), F(-0x1.23456p3));
  test<"{:$>14LA}">(SV("$$$-1#23456P+3"), F(-0x1.23456p3));
  test<"{:$^14LA}">(SV("$-1#23456P+3$$"), F(-0x1.23456p3));
  test<"{:014LA}">(SV("-0001#23456P+3"), F(-0x1.23456p3));

  test<"{:$<13LA}">(SV("1.23456P+3$$$"), en_US, F(0x1.23456p3));
  test<"{:$>13LA}">(SV("$$$1.23456P+3"), en_US, F(0x1.23456p3));
  test<"{:$^13LA}">(SV("$1.23456P+3$$"), en_US, F(0x1.23456p3));
  test<"{:013LA}">(SV("0001.23456P+3"), en_US, F(0x1.23456p3));
  test<"{:$<14LA}">(SV("-1.23456P+3$$$"), en_US, F(-0x1.23456p3));
  test<"{:$>14LA}">(SV("$$$-1.23456P+3"), en_US, F(-0x1.23456p3));
  test<"{:$^14LA}">(SV("$-1.23456P+3$$"), en_US, F(-0x1.23456p3));
  test<"{:014LA}">(SV("-0001.23456P+3"), en_US, F(-0x1.23456p3));

  std::locale::global(en_US);
  test<"{:$<13LA}">(SV("1#23456P+3$$$"), loc, F(0x1.23456p3));
  test<"{:$>13LA}">(SV("$$$1#23456P+3"), loc, F(0x1.23456p3));
  test<"{:$^13LA}">(SV("$1#23456P+3$$"), loc, F(0x1.23456p3));
  test<"{:013LA}">(SV("0001#23456P+3"), loc, F(0x1.23456p3));
  test<"{:$<14LA}">(SV("-1#23456P+3$$$"), loc, F(-0x1.23456p3));
  test<"{:$>14LA}">(SV("$$$-1#23456P+3"), loc, F(-0x1.23456p3));
  test<"{:$^14LA}">(SV("$-1#23456P+3$$"), loc, F(-0x1.23456p3));
  test<"{:014LA}">(SV("-0001#23456P+3"), loc, F(-0x1.23456p3));
}

template <class F, class CharT>
void test_floating_point_hex_lower_case_precision() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test<"{:.6La}">(SV("1.234560p-3"), F(0x1.23456p-3));
  test<"{:.6La}">(SV("1.234560p-2"), F(0x1.23456p-2));
  test<"{:.6La}">(SV("1.234560p-1"), F(0x1.23456p-1));
  test<"{:.6La}">(SV("1.234560p+0"), F(0x1.23456p0));
  test<"{:.6La}">(SV("1.234560p+1"), F(0x1.23456p+1));
  test<"{:.6La}">(SV("1.234560p+2"), F(0x1.23456p+2));
  test<"{:.6La}">(SV("1.234560p+3"), F(0x1.23456p+3));
  test<"{:.6La}">(SV("1.234560p+20"), F(0x1.23456p+20));

  std::locale::global(loc);
  test<"{:.6La}">(SV("1#234560p-3"), F(0x1.23456p-3));
  test<"{:.6La}">(SV("1#234560p-2"), F(0x1.23456p-2));
  test<"{:.6La}">(SV("1#234560p-1"), F(0x1.23456p-1));
  test<"{:.6La}">(SV("1#234560p+0"), F(0x1.23456p0));
  test<"{:.6La}">(SV("1#234560p+1"), F(0x1.23456p+1));
  test<"{:.6La}">(SV("1#234560p+2"), F(0x1.23456p+2));
  test<"{:.6La}">(SV("1#234560p+3"), F(0x1.23456p+3));
  test<"{:.6La}">(SV("1#234560p+20"), F(0x1.23456p+20));

  test<"{:.6La}">(SV("1.234560p-3"), en_US, F(0x1.23456p-3));
  test<"{:.6La}">(SV("1.234560p-2"), en_US, F(0x1.23456p-2));
  test<"{:.6La}">(SV("1.234560p-1"), en_US, F(0x1.23456p-1));
  test<"{:.6La}">(SV("1.234560p+0"), en_US, F(0x1.23456p0));
  test<"{:.6La}">(SV("1.234560p+1"), en_US, F(0x1.23456p+1));
  test<"{:.6La}">(SV("1.234560p+2"), en_US, F(0x1.23456p+2));
  test<"{:.6La}">(SV("1.234560p+3"), en_US, F(0x1.23456p+3));
  test<"{:.6La}">(SV("1.234560p+20"), en_US, F(0x1.23456p+20));

  std::locale::global(en_US);
  test<"{:.6La}">(SV("1#234560p-3"), loc, F(0x1.23456p-3));
  test<"{:.6La}">(SV("1#234560p-2"), loc, F(0x1.23456p-2));
  test<"{:.6La}">(SV("1#234560p-1"), loc, F(0x1.23456p-1));
  test<"{:.6La}">(SV("1#234560p+0"), loc, F(0x1.23456p0));
  test<"{:.6La}">(SV("1#234560p+1"), loc, F(0x1.23456p+1));
  test<"{:.6La}">(SV("1#234560p+2"), loc, F(0x1.23456p+2));
  test<"{:.6La}">(SV("1#234560p+3"), loc, F(0x1.23456p+3));
  test<"{:.6La}">(SV("1#234560p+20"), loc, F(0x1.23456p+20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test<"{:$<14.6La}">(SV("1.234560p+3$$$"), F(0x1.23456p3));
  test<"{:$>14.6La}">(SV("$$$1.234560p+3"), F(0x1.23456p3));
  test<"{:$^14.6La}">(SV("$1.234560p+3$$"), F(0x1.23456p3));
  test<"{:014.6La}">(SV("0001.234560p+3"), F(0x1.23456p3));
  test<"{:$<15.6La}">(SV("-1.234560p+3$$$"), F(-0x1.23456p3));
  test<"{:$>15.6La}">(SV("$$$-1.234560p+3"), F(-0x1.23456p3));
  test<"{:$^15.6La}">(SV("$-1.234560p+3$$"), F(-0x1.23456p3));
  test<"{:015.6La}">(SV("-0001.234560p+3"), F(-0x1.23456p3));

  std::locale::global(loc);
  test<"{:$<14.6La}">(SV("1#234560p+3$$$"), F(0x1.23456p3));
  test<"{:$>14.6La}">(SV("$$$1#234560p+3"), F(0x1.23456p3));
  test<"{:$^14.6La}">(SV("$1#234560p+3$$"), F(0x1.23456p3));
  test<"{:014.6La}">(SV("0001#234560p+3"), F(0x1.23456p3));
  test<"{:$<15.6La}">(SV("-1#234560p+3$$$"), F(-0x1.23456p3));
  test<"{:$>15.6La}">(SV("$$$-1#234560p+3"), F(-0x1.23456p3));
  test<"{:$^15.6La}">(SV("$-1#234560p+3$$"), F(-0x1.23456p3));
  test<"{:015.6La}">(SV("-0001#234560p+3"), F(-0x1.23456p3));

  test<"{:$<14.6La}">(SV("1.234560p+3$$$"), en_US, F(0x1.23456p3));
  test<"{:$>14.6La}">(SV("$$$1.234560p+3"), en_US, F(0x1.23456p3));
  test<"{:$^14.6La}">(SV("$1.234560p+3$$"), en_US, F(0x1.23456p3));
  test<"{:014.6La}">(SV("0001.234560p+3"), en_US, F(0x1.23456p3));
  test<"{:$<15.6La}">(SV("-1.234560p+3$$$"), en_US, F(-0x1.23456p3));
  test<"{:$>15.6La}">(SV("$$$-1.234560p+3"), en_US, F(-0x1.23456p3));
  test<"{:$^15.6La}">(SV("$-1.234560p+3$$"), en_US, F(-0x1.23456p3));
  test<"{:015.6La}">(SV("-0001.234560p+3"), en_US, F(-0x1.23456p3));

  std::locale::global(en_US);
  test<"{:$<14.6La}">(SV("1#234560p+3$$$"), loc, F(0x1.23456p3));
  test<"{:$>14.6La}">(SV("$$$1#234560p+3"), loc, F(0x1.23456p3));
  test<"{:$^14.6La}">(SV("$1#234560p+3$$"), loc, F(0x1.23456p3));
  test<"{:014.6La}">(SV("0001#234560p+3"), loc, F(0x1.23456p3));
  test<"{:$<15.6La}">(SV("-1#234560p+3$$$"), loc, F(-0x1.23456p3));
  test<"{:$>15.6La}">(SV("$$$-1#234560p+3"), loc, F(-0x1.23456p3));
  test<"{:$^15.6La}">(SV("$-1#234560p+3$$"), loc, F(-0x1.23456p3));
  test<"{:015.6La}">(SV("-0001#234560p+3"), loc, F(-0x1.23456p3));
}

template <class F, class CharT>
void test_floating_point_hex_upper_case_precision() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test<"{:.6LA}">(SV("1.234560P-3"), F(0x1.23456p-3));
  test<"{:.6LA}">(SV("1.234560P-2"), F(0x1.23456p-2));
  test<"{:.6LA}">(SV("1.234560P-1"), F(0x1.23456p-1));
  test<"{:.6LA}">(SV("1.234560P+0"), F(0x1.23456p0));
  test<"{:.6LA}">(SV("1.234560P+1"), F(0x1.23456p+1));
  test<"{:.6LA}">(SV("1.234560P+2"), F(0x1.23456p+2));
  test<"{:.6LA}">(SV("1.234560P+3"), F(0x1.23456p+3));
  test<"{:.6LA}">(SV("1.234560P+20"), F(0x1.23456p+20));

  std::locale::global(loc);
  test<"{:.6LA}">(SV("1#234560P-3"), F(0x1.23456p-3));
  test<"{:.6LA}">(SV("1#234560P-2"), F(0x1.23456p-2));
  test<"{:.6LA}">(SV("1#234560P-1"), F(0x1.23456p-1));
  test<"{:.6LA}">(SV("1#234560P+0"), F(0x1.23456p0));
  test<"{:.6LA}">(SV("1#234560P+1"), F(0x1.23456p+1));
  test<"{:.6LA}">(SV("1#234560P+2"), F(0x1.23456p+2));
  test<"{:.6LA}">(SV("1#234560P+3"), F(0x1.23456p+3));
  test<"{:.6LA}">(SV("1#234560P+20"), F(0x1.23456p+20));

  test<"{:.6LA}">(SV("1.234560P-3"), en_US, F(0x1.23456p-3));
  test<"{:.6LA}">(SV("1.234560P-2"), en_US, F(0x1.23456p-2));
  test<"{:.6LA}">(SV("1.234560P-1"), en_US, F(0x1.23456p-1));
  test<"{:.6LA}">(SV("1.234560P+0"), en_US, F(0x1.23456p0));
  test<"{:.6LA}">(SV("1.234560P+1"), en_US, F(0x1.23456p+1));
  test<"{:.6LA}">(SV("1.234560P+2"), en_US, F(0x1.23456p+2));
  test<"{:.6LA}">(SV("1.234560P+3"), en_US, F(0x1.23456p+3));
  test<"{:.6LA}">(SV("1.234560P+20"), en_US, F(0x1.23456p+20));

  std::locale::global(en_US);
  test<"{:.6LA}">(SV("1#234560P-3"), loc, F(0x1.23456p-3));
  test<"{:.6LA}">(SV("1#234560P-2"), loc, F(0x1.23456p-2));
  test<"{:.6LA}">(SV("1#234560P-1"), loc, F(0x1.23456p-1));
  test<"{:.6LA}">(SV("1#234560P+0"), loc, F(0x1.23456p0));
  test<"{:.6LA}">(SV("1#234560P+1"), loc, F(0x1.23456p+1));
  test<"{:.6LA}">(SV("1#234560P+2"), loc, F(0x1.23456p+2));
  test<"{:.6LA}">(SV("1#234560P+3"), loc, F(0x1.23456p+3));
  test<"{:.6LA}">(SV("1#234560P+20"), loc, F(0x1.23456p+20));

  // *** Fill, align, zero Padding ***
  std::locale::global(en_US);
  test<"{:$<14.6LA}">(SV("1.234560P+3$$$"), F(0x1.23456p3));
  test<"{:$>14.6LA}">(SV("$$$1.234560P+3"), F(0x1.23456p3));
  test<"{:$^14.6LA}">(SV("$1.234560P+3$$"), F(0x1.23456p3));
  test<"{:014.6LA}">(SV("0001.234560P+3"), F(0x1.23456p3));
  test<"{:$<15.6LA}">(SV("-1.234560P+3$$$"), F(-0x1.23456p3));
  test<"{:$>15.6LA}">(SV("$$$-1.234560P+3"), F(-0x1.23456p3));
  test<"{:$^15.6LA}">(SV("$-1.234560P+3$$"), F(-0x1.23456p3));
  test<"{:015.6LA}">(SV("-0001.234560P+3"), F(-0x1.23456p3));

  std::locale::global(loc);
  test<"{:$<14.6LA}">(SV("1#234560P+3$$$"), F(0x1.23456p3));
  test<"{:$>14.6LA}">(SV("$$$1#234560P+3"), F(0x1.23456p3));
  test<"{:$^14.6LA}">(SV("$1#234560P+3$$"), F(0x1.23456p3));
  test<"{:014.6LA}">(SV("0001#234560P+3"), F(0x1.23456p3));
  test<"{:$<15.6LA}">(SV("-1#234560P+3$$$"), F(-0x1.23456p3));
  test<"{:$>15.6LA}">(SV("$$$-1#234560P+3"), F(-0x1.23456p3));
  test<"{:$^15.6LA}">(SV("$-1#234560P+3$$"), F(-0x1.23456p3));
  test<"{:015.6LA}">(SV("-0001#234560P+3"), F(-0x1.23456p3));

  test<"{:$<14.6LA}">(SV("1.234560P+3$$$"), en_US, F(0x1.23456p3));
  test<"{:$>14.6LA}">(SV("$$$1.234560P+3"), en_US, F(0x1.23456p3));
  test<"{:$^14.6LA}">(SV("$1.234560P+3$$"), en_US, F(0x1.23456p3));
  test<"{:014.6LA}">(SV("0001.234560P+3"), en_US, F(0x1.23456p3));
  test<"{:$<15.6LA}">(SV("-1.234560P+3$$$"), en_US, F(-0x1.23456p3));
  test<"{:$>15.6LA}">(SV("$$$-1.234560P+3"), en_US, F(-0x1.23456p3));
  test<"{:$^15.6LA}">(SV("$-1.234560P+3$$"), en_US, F(-0x1.23456p3));
  test<"{:015.6LA}">(SV("-0001.234560P+3"), en_US, F(-0x1.23456p3));

  std::locale::global(en_US);
  test<"{:$<14.6LA}">(SV("1#234560P+3$$$"), loc, F(0x1.23456p3));
  test<"{:$>14.6LA}">(SV("$$$1#234560P+3"), loc, F(0x1.23456p3));
  test<"{:$^14.6LA}">(SV("$1#234560P+3$$"), loc, F(0x1.23456p3));
  test<"{:014.6LA}">(SV("0001#234560P+3"), loc, F(0x1.23456p3));
  test<"{:$<15.6LA}">(SV("-1#234560P+3$$$"), loc, F(-0x1.23456p3));
  test<"{:$>15.6LA}">(SV("$$$-1#234560P+3"), loc, F(-0x1.23456p3));
  test<"{:$^15.6LA}">(SV("$-1#234560P+3$$"), loc, F(-0x1.23456p3));
  test<"{:015.6LA}">(SV("-0001#234560P+3"), loc, F(-0x1.23456p3));
}

template <class F, class CharT>
void test_floating_point_scientific_lower_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test<"{:.6Le}">(SV("1.234567e-03"), F(1.234567e-3));
  test<"{:.6Le}">(SV("1.234567e-02"), F(1.234567e-2));
  test<"{:.6Le}">(SV("1.234567e-01"), F(1.234567e-1));
  test<"{:.6Le}">(SV("1.234567e+00"), F(1.234567e0));
  test<"{:.6Le}">(SV("1.234567e+01"), F(1.234567e1));
  test<"{:.6Le}">(SV("1.234567e+02"), F(1.234567e2));
  test<"{:.6Le}">(SV("1.234567e+03"), F(1.234567e3));
  test<"{:.6Le}">(SV("1.234567e+20"), F(1.234567e20));
  test<"{:.6Le}">(SV("-1.234567e-03"), F(-1.234567e-3));
  test<"{:.6Le}">(SV("-1.234567e-02"), F(-1.234567e-2));
  test<"{:.6Le}">(SV("-1.234567e-01"), F(-1.234567e-1));
  test<"{:.6Le}">(SV("-1.234567e+00"), F(-1.234567e0));
  test<"{:.6Le}">(SV("-1.234567e+01"), F(-1.234567e1));
  test<"{:.6Le}">(SV("-1.234567e+02"), F(-1.234567e2));
  test<"{:.6Le}">(SV("-1.234567e+03"), F(-1.234567e3));
  test<"{:.6Le}">(SV("-1.234567e+20"), F(-1.234567e20));

  std::locale::global(loc);
  test<"{:.6Le}">(SV("1#234567e-03"), F(1.234567e-3));
  test<"{:.6Le}">(SV("1#234567e-02"), F(1.234567e-2));
  test<"{:.6Le}">(SV("1#234567e-01"), F(1.234567e-1));
  test<"{:.6Le}">(SV("1#234567e+00"), F(1.234567e0));
  test<"{:.6Le}">(SV("1#234567e+01"), F(1.234567e1));
  test<"{:.6Le}">(SV("1#234567e+02"), F(1.234567e2));
  test<"{:.6Le}">(SV("1#234567e+03"), F(1.234567e3));
  test<"{:.6Le}">(SV("1#234567e+20"), F(1.234567e20));
  test<"{:.6Le}">(SV("-1#234567e-03"), F(-1.234567e-3));
  test<"{:.6Le}">(SV("-1#234567e-02"), F(-1.234567e-2));
  test<"{:.6Le}">(SV("-1#234567e-01"), F(-1.234567e-1));
  test<"{:.6Le}">(SV("-1#234567e+00"), F(-1.234567e0));
  test<"{:.6Le}">(SV("-1#234567e+01"), F(-1.234567e1));
  test<"{:.6Le}">(SV("-1#234567e+02"), F(-1.234567e2));
  test<"{:.6Le}">(SV("-1#234567e+03"), F(-1.234567e3));
  test<"{:.6Le}">(SV("-1#234567e+20"), F(-1.234567e20));

  test<"{:.6Le}">(SV("1.234567e-03"), en_US, F(1.234567e-3));
  test<"{:.6Le}">(SV("1.234567e-02"), en_US, F(1.234567e-2));
  test<"{:.6Le}">(SV("1.234567e-01"), en_US, F(1.234567e-1));
  test<"{:.6Le}">(SV("1.234567e+00"), en_US, F(1.234567e0));
  test<"{:.6Le}">(SV("1.234567e+01"), en_US, F(1.234567e1));
  test<"{:.6Le}">(SV("1.234567e+02"), en_US, F(1.234567e2));
  test<"{:.6Le}">(SV("1.234567e+03"), en_US, F(1.234567e3));
  test<"{:.6Le}">(SV("1.234567e+20"), en_US, F(1.234567e20));
  test<"{:.6Le}">(SV("-1.234567e-03"), en_US, F(-1.234567e-3));
  test<"{:.6Le}">(SV("-1.234567e-02"), en_US, F(-1.234567e-2));
  test<"{:.6Le}">(SV("-1.234567e-01"), en_US, F(-1.234567e-1));
  test<"{:.6Le}">(SV("-1.234567e+00"), en_US, F(-1.234567e0));
  test<"{:.6Le}">(SV("-1.234567e+01"), en_US, F(-1.234567e1));
  test<"{:.6Le}">(SV("-1.234567e+02"), en_US, F(-1.234567e2));
  test<"{:.6Le}">(SV("-1.234567e+03"), en_US, F(-1.234567e3));
  test<"{:.6Le}">(SV("-1.234567e+20"), en_US, F(-1.234567e20));

  std::locale::global(en_US);
  test<"{:.6Le}">(SV("1#234567e-03"), loc, F(1.234567e-3));
  test<"{:.6Le}">(SV("1#234567e-02"), loc, F(1.234567e-2));
  test<"{:.6Le}">(SV("1#234567e-01"), loc, F(1.234567e-1));
  test<"{:.6Le}">(SV("1#234567e+00"), loc, F(1.234567e0));
  test<"{:.6Le}">(SV("1#234567e+01"), loc, F(1.234567e1));
  test<"{:.6Le}">(SV("1#234567e+02"), loc, F(1.234567e2));
  test<"{:.6Le}">(SV("1#234567e+03"), loc, F(1.234567e3));
  test<"{:.6Le}">(SV("1#234567e+20"), loc, F(1.234567e20));
  test<"{:.6Le}">(SV("-1#234567e-03"), loc, F(-1.234567e-3));
  test<"{:.6Le}">(SV("-1#234567e-02"), loc, F(-1.234567e-2));
  test<"{:.6Le}">(SV("-1#234567e-01"), loc, F(-1.234567e-1));
  test<"{:.6Le}">(SV("-1#234567e+00"), loc, F(-1.234567e0));
  test<"{:.6Le}">(SV("-1#234567e+01"), loc, F(-1.234567e1));
  test<"{:.6Le}">(SV("-1#234567e+02"), loc, F(-1.234567e2));
  test<"{:.6Le}">(SV("-1#234567e+03"), loc, F(-1.234567e3));
  test<"{:.6Le}">(SV("-1#234567e+20"), loc, F(-1.234567e20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test<"{:$<15.6Le}">(SV("1.234567e+03$$$"), F(1.234567e3));
  test<"{:$>15.6Le}">(SV("$$$1.234567e+03"), F(1.234567e3));
  test<"{:$^15.6Le}">(SV("$1.234567e+03$$"), F(1.234567e3));
  test<"{:015.6Le}">(SV("0001.234567e+03"), F(1.234567e3));
  test<"{:$<16.6Le}">(SV("-1.234567e+03$$$"), F(-1.234567e3));
  test<"{:$>16.6Le}">(SV("$$$-1.234567e+03"), F(-1.234567e3));
  test<"{:$^16.6Le}">(SV("$-1.234567e+03$$"), F(-1.234567e3));
  test<"{:016.6Le}">(SV("-0001.234567e+03"), F(-1.234567e3));

  std::locale::global(loc);
  test<"{:$<15.6Le}">(SV("1#234567e+03$$$"), F(1.234567e3));
  test<"{:$>15.6Le}">(SV("$$$1#234567e+03"), F(1.234567e3));
  test<"{:$^15.6Le}">(SV("$1#234567e+03$$"), F(1.234567e3));
  test<"{:015.6Le}">(SV("0001#234567e+03"), F(1.234567e3));
  test<"{:$<16.6Le}">(SV("-1#234567e+03$$$"), F(-1.234567e3));
  test<"{:$>16.6Le}">(SV("$$$-1#234567e+03"), F(-1.234567e3));
  test<"{:$^16.6Le}">(SV("$-1#234567e+03$$"), F(-1.234567e3));
  test<"{:016.6Le}">(SV("-0001#234567e+03"), F(-1.234567e3));

  test<"{:$<15.6Le}">(SV("1.234567e+03$$$"), en_US, F(1.234567e3));
  test<"{:$>15.6Le}">(SV("$$$1.234567e+03"), en_US, F(1.234567e3));
  test<"{:$^15.6Le}">(SV("$1.234567e+03$$"), en_US, F(1.234567e3));
  test<"{:015.6Le}">(SV("0001.234567e+03"), en_US, F(1.234567e3));
  test<"{:$<16.6Le}">(SV("-1.234567e+03$$$"), en_US, F(-1.234567e3));
  test<"{:$>16.6Le}">(SV("$$$-1.234567e+03"), en_US, F(-1.234567e3));
  test<"{:$^16.6Le}">(SV("$-1.234567e+03$$"), en_US, F(-1.234567e3));
  test<"{:016.6Le}">(SV("-0001.234567e+03"), en_US, F(-1.234567e3));

  std::locale::global(en_US);
  test<"{:$<15.6Le}">(SV("1#234567e+03$$$"), loc, F(1.234567e3));
  test<"{:$>15.6Le}">(SV("$$$1#234567e+03"), loc, F(1.234567e3));
  test<"{:$^15.6Le}">(SV("$1#234567e+03$$"), loc, F(1.234567e3));
  test<"{:015.6Le}">(SV("0001#234567e+03"), loc, F(1.234567e3));
  test<"{:$<16.6Le}">(SV("-1#234567e+03$$$"), loc, F(-1.234567e3));
  test<"{:$>16.6Le}">(SV("$$$-1#234567e+03"), loc, F(-1.234567e3));
  test<"{:$^16.6Le}">(SV("$-1#234567e+03$$"), loc, F(-1.234567e3));
  test<"{:016.6Le}">(SV("-0001#234567e+03"), loc, F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_scientific_upper_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test<"{:.6LE}">(SV("1.234567E-03"), F(1.234567e-3));
  test<"{:.6LE}">(SV("1.234567E-02"), F(1.234567e-2));
  test<"{:.6LE}">(SV("1.234567E-01"), F(1.234567e-1));
  test<"{:.6LE}">(SV("1.234567E+00"), F(1.234567e0));
  test<"{:.6LE}">(SV("1.234567E+01"), F(1.234567e1));
  test<"{:.6LE}">(SV("1.234567E+02"), F(1.234567e2));
  test<"{:.6LE}">(SV("1.234567E+03"), F(1.234567e3));
  test<"{:.6LE}">(SV("1.234567E+20"), F(1.234567e20));
  test<"{:.6LE}">(SV("-1.234567E-03"), F(-1.234567e-3));
  test<"{:.6LE}">(SV("-1.234567E-02"), F(-1.234567e-2));
  test<"{:.6LE}">(SV("-1.234567E-01"), F(-1.234567e-1));
  test<"{:.6LE}">(SV("-1.234567E+00"), F(-1.234567e0));
  test<"{:.6LE}">(SV("-1.234567E+01"), F(-1.234567e1));
  test<"{:.6LE}">(SV("-1.234567E+02"), F(-1.234567e2));
  test<"{:.6LE}">(SV("-1.234567E+03"), F(-1.234567e3));
  test<"{:.6LE}">(SV("-1.234567E+20"), F(-1.234567e20));

  std::locale::global(loc);
  test<"{:.6LE}">(SV("1#234567E-03"), F(1.234567e-3));
  test<"{:.6LE}">(SV("1#234567E-02"), F(1.234567e-2));
  test<"{:.6LE}">(SV("1#234567E-01"), F(1.234567e-1));
  test<"{:.6LE}">(SV("1#234567E+00"), F(1.234567e0));
  test<"{:.6LE}">(SV("1#234567E+01"), F(1.234567e1));
  test<"{:.6LE}">(SV("1#234567E+02"), F(1.234567e2));
  test<"{:.6LE}">(SV("1#234567E+03"), F(1.234567e3));
  test<"{:.6LE}">(SV("1#234567E+20"), F(1.234567e20));
  test<"{:.6LE}">(SV("-1#234567E-03"), F(-1.234567e-3));
  test<"{:.6LE}">(SV("-1#234567E-02"), F(-1.234567e-2));
  test<"{:.6LE}">(SV("-1#234567E-01"), F(-1.234567e-1));
  test<"{:.6LE}">(SV("-1#234567E+00"), F(-1.234567e0));
  test<"{:.6LE}">(SV("-1#234567E+01"), F(-1.234567e1));
  test<"{:.6LE}">(SV("-1#234567E+02"), F(-1.234567e2));
  test<"{:.6LE}">(SV("-1#234567E+03"), F(-1.234567e3));
  test<"{:.6LE}">(SV("-1#234567E+20"), F(-1.234567e20));

  test<"{:.6LE}">(SV("1.234567E-03"), en_US, F(1.234567e-3));
  test<"{:.6LE}">(SV("1.234567E-02"), en_US, F(1.234567e-2));
  test<"{:.6LE}">(SV("1.234567E-01"), en_US, F(1.234567e-1));
  test<"{:.6LE}">(SV("1.234567E+00"), en_US, F(1.234567e0));
  test<"{:.6LE}">(SV("1.234567E+01"), en_US, F(1.234567e1));
  test<"{:.6LE}">(SV("1.234567E+02"), en_US, F(1.234567e2));
  test<"{:.6LE}">(SV("1.234567E+03"), en_US, F(1.234567e3));
  test<"{:.6LE}">(SV("1.234567E+20"), en_US, F(1.234567e20));
  test<"{:.6LE}">(SV("-1.234567E-03"), en_US, F(-1.234567e-3));
  test<"{:.6LE}">(SV("-1.234567E-02"), en_US, F(-1.234567e-2));
  test<"{:.6LE}">(SV("-1.234567E-01"), en_US, F(-1.234567e-1));
  test<"{:.6LE}">(SV("-1.234567E+00"), en_US, F(-1.234567e0));
  test<"{:.6LE}">(SV("-1.234567E+01"), en_US, F(-1.234567e1));
  test<"{:.6LE}">(SV("-1.234567E+02"), en_US, F(-1.234567e2));
  test<"{:.6LE}">(SV("-1.234567E+03"), en_US, F(-1.234567e3));
  test<"{:.6LE}">(SV("-1.234567E+20"), en_US, F(-1.234567e20));

  std::locale::global(en_US);
  test<"{:.6LE}">(SV("1#234567E-03"), loc, F(1.234567e-3));
  test<"{:.6LE}">(SV("1#234567E-02"), loc, F(1.234567e-2));
  test<"{:.6LE}">(SV("1#234567E-01"), loc, F(1.234567e-1));
  test<"{:.6LE}">(SV("1#234567E+00"), loc, F(1.234567e0));
  test<"{:.6LE}">(SV("1#234567E+01"), loc, F(1.234567e1));
  test<"{:.6LE}">(SV("1#234567E+02"), loc, F(1.234567e2));
  test<"{:.6LE}">(SV("1#234567E+03"), loc, F(1.234567e3));
  test<"{:.6LE}">(SV("1#234567E+20"), loc, F(1.234567e20));
  test<"{:.6LE}">(SV("-1#234567E-03"), loc, F(-1.234567e-3));
  test<"{:.6LE}">(SV("-1#234567E-02"), loc, F(-1.234567e-2));
  test<"{:.6LE}">(SV("-1#234567E-01"), loc, F(-1.234567e-1));
  test<"{:.6LE}">(SV("-1#234567E+00"), loc, F(-1.234567e0));
  test<"{:.6LE}">(SV("-1#234567E+01"), loc, F(-1.234567e1));
  test<"{:.6LE}">(SV("-1#234567E+02"), loc, F(-1.234567e2));
  test<"{:.6LE}">(SV("-1#234567E+03"), loc, F(-1.234567e3));
  test<"{:.6LE}">(SV("-1#234567E+20"), loc, F(-1.234567e20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test<"{:$<15.6LE}">(SV("1.234567E+03$$$"), F(1.234567e3));
  test<"{:$>15.6LE}">(SV("$$$1.234567E+03"), F(1.234567e3));
  test<"{:$^15.6LE}">(SV("$1.234567E+03$$"), F(1.234567e3));
  test<"{:015.6LE}">(SV("0001.234567E+03"), F(1.234567e3));
  test<"{:$<16.6LE}">(SV("-1.234567E+03$$$"), F(-1.234567e3));
  test<"{:$>16.6LE}">(SV("$$$-1.234567E+03"), F(-1.234567e3));
  test<"{:$^16.6LE}">(SV("$-1.234567E+03$$"), F(-1.234567e3));
  test<"{:016.6LE}">(SV("-0001.234567E+03"), F(-1.234567e3));

  std::locale::global(loc);
  test<"{:$<15.6LE}">(SV("1#234567E+03$$$"), F(1.234567e3));
  test<"{:$>15.6LE}">(SV("$$$1#234567E+03"), F(1.234567e3));
  test<"{:$^15.6LE}">(SV("$1#234567E+03$$"), F(1.234567e3));
  test<"{:015.6LE}">(SV("0001#234567E+03"), F(1.234567e3));
  test<"{:$<16.6LE}">(SV("-1#234567E+03$$$"), F(-1.234567e3));
  test<"{:$>16.6LE}">(SV("$$$-1#234567E+03"), F(-1.234567e3));
  test<"{:$^16.6LE}">(SV("$-1#234567E+03$$"), F(-1.234567e3));
  test<"{:016.6LE}">(SV("-0001#234567E+03"), F(-1.234567e3));

  test<"{:$<15.6LE}">(SV("1.234567E+03$$$"), en_US, F(1.234567e3));
  test<"{:$>15.6LE}">(SV("$$$1.234567E+03"), en_US, F(1.234567e3));
  test<"{:$^15.6LE}">(SV("$1.234567E+03$$"), en_US, F(1.234567e3));
  test<"{:015.6LE}">(SV("0001.234567E+03"), en_US, F(1.234567e3));
  test<"{:$<16.6LE}">(SV("-1.234567E+03$$$"), en_US, F(-1.234567e3));
  test<"{:$>16.6LE}">(SV("$$$-1.234567E+03"), en_US, F(-1.234567e3));
  test<"{:$^16.6LE}">(SV("$-1.234567E+03$$"), en_US, F(-1.234567e3));
  test<"{:016.6LE}">(SV("-0001.234567E+03"), en_US, F(-1.234567e3));

  std::locale::global(en_US);
  test<"{:$<15.6LE}">(SV("1#234567E+03$$$"), loc, F(1.234567e3));
  test<"{:$>15.6LE}">(SV("$$$1#234567E+03"), loc, F(1.234567e3));
  test<"{:$^15.6LE}">(SV("$1#234567E+03$$"), loc, F(1.234567e3));
  test<"{:015.6LE}">(SV("0001#234567E+03"), loc, F(1.234567e3));
  test<"{:$<16.6LE}">(SV("-1#234567E+03$$$"), loc, F(-1.234567e3));
  test<"{:$>16.6LE}">(SV("$$$-1#234567E+03"), loc, F(-1.234567e3));
  test<"{:$^16.6LE}">(SV("$-1#234567E+03$$"), loc, F(-1.234567e3));
  test<"{:016.6LE}">(SV("-0001#234567E+03"), loc, F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_fixed_lower_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test<"{:.6Lf}">(SV("0.000001"), F(1.234567e-6));
  test<"{:.6Lf}">(SV("0.000012"), F(1.234567e-5));
  test<"{:.6Lf}">(SV("0.000123"), F(1.234567e-4));
  test<"{:.6Lf}">(SV("0.001235"), F(1.234567e-3));
  test<"{:.6Lf}">(SV("0.012346"), F(1.234567e-2));
  test<"{:.6Lf}">(SV("0.123457"), F(1.234567e-1));
  test<"{:.6Lf}">(SV("1.234567"), F(1.234567e0));
  test<"{:.6Lf}">(SV("12.345670"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("123.456700"), F(1.234567e2));
    test<"{:.6Lf}">(SV("1,234.567000"), F(1.234567e3));
    test<"{:.6Lf}">(SV("12,345.670000"), F(1.234567e4));
    test<"{:.6Lf}">(SV("123,456.700000"), F(1.234567e5));
    test<"{:.6Lf}">(SV("1,234,567.000000"), F(1.234567e6));
    test<"{:.6Lf}">(SV("12,345,670.000000"), F(1.234567e7));
    test<"{:.6Lf}">(SV("123,456,700,000,000,000,000.000000"), F(1.234567e20));
  }
  test<"{:.6Lf}">(SV("-0.000001"), F(-1.234567e-6));
  test<"{:.6Lf}">(SV("-0.000012"), F(-1.234567e-5));
  test<"{:.6Lf}">(SV("-0.000123"), F(-1.234567e-4));
  test<"{:.6Lf}">(SV("-0.001235"), F(-1.234567e-3));
  test<"{:.6Lf}">(SV("-0.012346"), F(-1.234567e-2));
  test<"{:.6Lf}">(SV("-0.123457"), F(-1.234567e-1));
  test<"{:.6Lf}">(SV("-1.234567"), F(-1.234567e0));
  test<"{:.6Lf}">(SV("-12.345670"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("-123.456700"), F(-1.234567e2));
    test<"{:.6Lf}">(SV("-1,234.567000"), F(-1.234567e3));
    test<"{:.6Lf}">(SV("-12,345.670000"), F(-1.234567e4));
    test<"{:.6Lf}">(SV("-123,456.700000"), F(-1.234567e5));
    test<"{:.6Lf}">(SV("-1,234,567.000000"), F(-1.234567e6));
    test<"{:.6Lf}">(SV("-12,345,670.000000"), F(-1.234567e7));
    test<"{:.6Lf}">(SV("-123,456,700,000,000,000,000.000000"), F(-1.234567e20));
  }

  std::locale::global(loc);
  test<"{:.6Lf}">(SV("0#000001"), F(1.234567e-6));
  test<"{:.6Lf}">(SV("0#000012"), F(1.234567e-5));
  test<"{:.6Lf}">(SV("0#000123"), F(1.234567e-4));
  test<"{:.6Lf}">(SV("0#001235"), F(1.234567e-3));
  test<"{:.6Lf}">(SV("0#012346"), F(1.234567e-2));
  test<"{:.6Lf}">(SV("0#123457"), F(1.234567e-1));
  test<"{:.6Lf}">(SV("1#234567"), F(1.234567e0));
  test<"{:.6Lf}">(SV("1_2#345670"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("12_3#456700"), F(1.234567e2));
    test<"{:.6Lf}">(SV("1_23_4#567000"), F(1.234567e3));
    test<"{:.6Lf}">(SV("12_34_5#670000"), F(1.234567e4));
    test<"{:.6Lf}">(SV("123_45_6#700000"), F(1.234567e5));
    test<"{:.6Lf}">(SV("1_234_56_7#000000"), F(1.234567e6));
    test<"{:.6Lf}">(SV("12_345_67_0#000000"), F(1.234567e7));
    test<"{:.6Lf}">(SV("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), F(1.234567e20));
  }
  test<"{:.6Lf}">(SV("-0#000001"), F(-1.234567e-6));
  test<"{:.6Lf}">(SV("-0#000012"), F(-1.234567e-5));
  test<"{:.6Lf}">(SV("-0#000123"), F(-1.234567e-4));
  test<"{:.6Lf}">(SV("-0#001235"), F(-1.234567e-3));
  test<"{:.6Lf}">(SV("-0#012346"), F(-1.234567e-2));
  test<"{:.6Lf}">(SV("-0#123457"), F(-1.234567e-1));
  test<"{:.6Lf}">(SV("-1#234567"), F(-1.234567e0));
  test<"{:.6Lf}">(SV("-1_2#345670"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("-12_3#456700"), F(-1.234567e2));
    test<"{:.6Lf}">(SV("-1_23_4#567000"), F(-1.234567e3));
    test<"{:.6Lf}">(SV("-12_34_5#670000"), F(-1.234567e4));
    test<"{:.6Lf}">(SV("-123_45_6#700000"), F(-1.234567e5));
    test<"{:.6Lf}">(SV("-1_234_56_7#000000"), F(-1.234567e6));
    test<"{:.6Lf}">(SV("-12_345_67_0#000000"), F(-1.234567e7));
    test<"{:.6Lf}">(SV("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), F(-1.234567e20));
  }

  test<"{:.6Lf}">(SV("0.000001"), en_US, F(1.234567e-6));
  test<"{:.6Lf}">(SV("0.000012"), en_US, F(1.234567e-5));
  test<"{:.6Lf}">(SV("0.000123"), en_US, F(1.234567e-4));
  test<"{:.6Lf}">(SV("0.001235"), en_US, F(1.234567e-3));
  test<"{:.6Lf}">(SV("0.012346"), en_US, F(1.234567e-2));
  test<"{:.6Lf}">(SV("0.123457"), en_US, F(1.234567e-1));
  test<"{:.6Lf}">(SV("1.234567"), en_US, F(1.234567e0));
  test<"{:.6Lf}">(SV("12.345670"), en_US, F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("123.456700"), en_US, F(1.234567e2));
    test<"{:.6Lf}">(SV("1,234.567000"), en_US, F(1.234567e3));
    test<"{:.6Lf}">(SV("12,345.670000"), en_US, F(1.234567e4));
    test<"{:.6Lf}">(SV("123,456.700000"), en_US, F(1.234567e5));
    test<"{:.6Lf}">(SV("1,234,567.000000"), en_US, F(1.234567e6));
    test<"{:.6Lf}">(SV("12,345,670.000000"), en_US, F(1.234567e7));
    test<"{:.6Lf}">(SV("123,456,700,000,000,000,000.000000"), en_US, F(1.234567e20));
  }
  test<"{:.6Lf}">(SV("-0.000001"), en_US, F(-1.234567e-6));
  test<"{:.6Lf}">(SV("-0.000012"), en_US, F(-1.234567e-5));
  test<"{:.6Lf}">(SV("-0.000123"), en_US, F(-1.234567e-4));
  test<"{:.6Lf}">(SV("-0.001235"), en_US, F(-1.234567e-3));
  test<"{:.6Lf}">(SV("-0.012346"), en_US, F(-1.234567e-2));
  test<"{:.6Lf}">(SV("-0.123457"), en_US, F(-1.234567e-1));
  test<"{:.6Lf}">(SV("-1.234567"), en_US, F(-1.234567e0));
  test<"{:.6Lf}">(SV("-12.345670"), en_US, F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("-123.456700"), en_US, F(-1.234567e2));
    test<"{:.6Lf}">(SV("-1,234.567000"), en_US, F(-1.234567e3));
    test<"{:.6Lf}">(SV("-12,345.670000"), en_US, F(-1.234567e4));
    test<"{:.6Lf}">(SV("-123,456.700000"), en_US, F(-1.234567e5));
    test<"{:.6Lf}">(SV("-1,234,567.000000"), en_US, F(-1.234567e6));
    test<"{:.6Lf}">(SV("-12,345,670.000000"), en_US, F(-1.234567e7));
    test<"{:.6Lf}">(SV("-123,456,700,000,000,000,000.000000"), en_US, F(-1.234567e20));
  }

  std::locale::global(en_US);
  test<"{:.6Lf}">(SV("0#000001"), loc, F(1.234567e-6));
  test<"{:.6Lf}">(SV("0#000012"), loc, F(1.234567e-5));
  test<"{:.6Lf}">(SV("0#000123"), loc, F(1.234567e-4));
  test<"{:.6Lf}">(SV("0#001235"), loc, F(1.234567e-3));
  test<"{:.6Lf}">(SV("0#012346"), loc, F(1.234567e-2));
  test<"{:.6Lf}">(SV("0#123457"), loc, F(1.234567e-1));
  test<"{:.6Lf}">(SV("1#234567"), loc, F(1.234567e0));
  test<"{:.6Lf}">(SV("1_2#345670"), loc, F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("12_3#456700"), loc, F(1.234567e2));
    test<"{:.6Lf}">(SV("1_23_4#567000"), loc, F(1.234567e3));
    test<"{:.6Lf}">(SV("12_34_5#670000"), loc, F(1.234567e4));
    test<"{:.6Lf}">(SV("123_45_6#700000"), loc, F(1.234567e5));
    test<"{:.6Lf}">(SV("1_234_56_7#000000"), loc, F(1.234567e6));
    test<"{:.6Lf}">(SV("12_345_67_0#000000"), loc, F(1.234567e7));
    test<"{:.6Lf}">(SV("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), loc, F(1.234567e20));
  }
  test<"{:.6Lf}">(SV("-0#000001"), loc, F(-1.234567e-6));
  test<"{:.6Lf}">(SV("-0#000012"), loc, F(-1.234567e-5));
  test<"{:.6Lf}">(SV("-0#000123"), loc, F(-1.234567e-4));
  test<"{:.6Lf}">(SV("-0#001235"), loc, F(-1.234567e-3));
  test<"{:.6Lf}">(SV("-0#012346"), loc, F(-1.234567e-2));
  test<"{:.6Lf}">(SV("-0#123457"), loc, F(-1.234567e-1));
  test<"{:.6Lf}">(SV("-1#234567"), loc, F(-1.234567e0));
  test<"{:.6Lf}">(SV("-1_2#345670"), loc, F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("-12_3#456700"), loc, F(-1.234567e2));
    test<"{:.6Lf}">(SV("-1_23_4#567000"), loc, F(-1.234567e3));
    test<"{:.6Lf}">(SV("-12_34_5#670000"), loc, F(-1.234567e4));
    test<"{:.6Lf}">(SV("-123_45_6#700000"), loc, F(-1.234567e5));
    test<"{:.6Lf}">(SV("-1_234_56_7#000000"), loc, F(-1.234567e6));
    test<"{:.6Lf}">(SV("-12_345_67_0#000000"), loc, F(-1.234567e7));
    test<"{:.6Lf}">(SV("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), loc, F(-1.234567e20));
  }

  // *** Fill, align, zero padding ***
  if constexpr (sizeof(F) > sizeof(float)) {
    std::locale::global(en_US);
    test<"{:$<15.6Lf}">(SV("1,234.567000$$$"), F(1.234567e3));
    test<"{:$>15.6Lf}">(SV("$$$1,234.567000"), F(1.234567e3));
    test<"{:$^15.6Lf}">(SV("$1,234.567000$$"), F(1.234567e3));
    test<"{:015.6Lf}">(SV("0001,234.567000"), F(1.234567e3));
    test<"{:$<16.6Lf}">(SV("-1,234.567000$$$"), F(-1.234567e3));
    test<"{:$>16.6Lf}">(SV("$$$-1,234.567000"), F(-1.234567e3));
    test<"{:$^16.6Lf}">(SV("$-1,234.567000$$"), F(-1.234567e3));
    test<"{:016.6Lf}">(SV("-0001,234.567000"), F(-1.234567e3));

    std::locale::global(loc);
    test<"{:$<16.6Lf}">(SV("1_23_4#567000$$$"), F(1.234567e3));
    test<"{:$>16.6Lf}">(SV("$$$1_23_4#567000"), F(1.234567e3));
    test<"{:$^16.6Lf}">(SV("$1_23_4#567000$$"), F(1.234567e3));
    test<"{:016.6Lf}">(SV("0001_23_4#567000"), F(1.234567e3));
    test<"{:$<17.6Lf}">(SV("-1_23_4#567000$$$"), F(-1.234567e3));
    test<"{:$>17.6Lf}">(SV("$$$-1_23_4#567000"), F(-1.234567e3));
    test<"{:$^17.6Lf}">(SV("$-1_23_4#567000$$"), F(-1.234567e3));
    test<"{:017.6Lf}">(SV("-0001_23_4#567000"), F(-1.234567e3));

    test<"{:$<15.6Lf}">(SV("1,234.567000$$$"), en_US, F(1.234567e3));
    test<"{:$>15.6Lf}">(SV("$$$1,234.567000"), en_US, F(1.234567e3));
    test<"{:$^15.6Lf}">(SV("$1,234.567000$$"), en_US, F(1.234567e3));
    test<"{:015.6Lf}">(SV("0001,234.567000"), en_US, F(1.234567e3));
    test<"{:$<16.6Lf}">(SV("-1,234.567000$$$"), en_US, F(-1.234567e3));
    test<"{:$>16.6Lf}">(SV("$$$-1,234.567000"), en_US, F(-1.234567e3));
    test<"{:$^16.6Lf}">(SV("$-1,234.567000$$"), en_US, F(-1.234567e3));
    test<"{:016.6Lf}">(SV("-0001,234.567000"), en_US, F(-1.234567e3));

    std::locale::global(en_US);
    test<"{:$<16.6Lf}">(SV("1_23_4#567000$$$"), loc, F(1.234567e3));
    test<"{:$>16.6Lf}">(SV("$$$1_23_4#567000"), loc, F(1.234567e3));
    test<"{:$^16.6Lf}">(SV("$1_23_4#567000$$"), loc, F(1.234567e3));
    test<"{:016.6Lf}">(SV("0001_23_4#567000"), loc, F(1.234567e3));
    test<"{:$<17.6Lf}">(SV("-1_23_4#567000$$$"), loc, F(-1.234567e3));
    test<"{:$>17.6Lf}">(SV("$$$-1_23_4#567000"), loc, F(-1.234567e3));
    test<"{:$^17.6Lf}">(SV("$-1_23_4#567000$$"), loc, F(-1.234567e3));
    test<"{:017.6Lf}">(SV("-0001_23_4#567000"), loc, F(-1.234567e3));
  }
}

template <class F, class CharT>
void test_floating_point_fixed_upper_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test<"{:.6Lf}">(SV("0.000001"), F(1.234567e-6));
  test<"{:.6Lf}">(SV("0.000012"), F(1.234567e-5));
  test<"{:.6Lf}">(SV("0.000123"), F(1.234567e-4));
  test<"{:.6Lf}">(SV("0.001235"), F(1.234567e-3));
  test<"{:.6Lf}">(SV("0.012346"), F(1.234567e-2));
  test<"{:.6Lf}">(SV("0.123457"), F(1.234567e-1));
  test<"{:.6Lf}">(SV("1.234567"), F(1.234567e0));
  test<"{:.6Lf}">(SV("12.345670"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("123.456700"), F(1.234567e2));
    test<"{:.6Lf}">(SV("1,234.567000"), F(1.234567e3));
    test<"{:.6Lf}">(SV("12,345.670000"), F(1.234567e4));
    test<"{:.6Lf}">(SV("123,456.700000"), F(1.234567e5));
    test<"{:.6Lf}">(SV("1,234,567.000000"), F(1.234567e6));
    test<"{:.6Lf}">(SV("12,345,670.000000"), F(1.234567e7));
    test<"{:.6Lf}">(SV("123,456,700,000,000,000,000.000000"), F(1.234567e20));
  }
  test<"{:.6Lf}">(SV("-0.000001"), F(-1.234567e-6));
  test<"{:.6Lf}">(SV("-0.000012"), F(-1.234567e-5));
  test<"{:.6Lf}">(SV("-0.000123"), F(-1.234567e-4));
  test<"{:.6Lf}">(SV("-0.001235"), F(-1.234567e-3));
  test<"{:.6Lf}">(SV("-0.012346"), F(-1.234567e-2));
  test<"{:.6Lf}">(SV("-0.123457"), F(-1.234567e-1));
  test<"{:.6Lf}">(SV("-1.234567"), F(-1.234567e0));
  test<"{:.6Lf}">(SV("-12.345670"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("-123.456700"), F(-1.234567e2));
    test<"{:.6Lf}">(SV("-1,234.567000"), F(-1.234567e3));
    test<"{:.6Lf}">(SV("-12,345.670000"), F(-1.234567e4));
    test<"{:.6Lf}">(SV("-123,456.700000"), F(-1.234567e5));
    test<"{:.6Lf}">(SV("-1,234,567.000000"), F(-1.234567e6));
    test<"{:.6Lf}">(SV("-12,345,670.000000"), F(-1.234567e7));
    test<"{:.6Lf}">(SV("-123,456,700,000,000,000,000.000000"), F(-1.234567e20));
  }

  std::locale::global(loc);
  test<"{:.6Lf}">(SV("0#000001"), F(1.234567e-6));
  test<"{:.6Lf}">(SV("0#000012"), F(1.234567e-5));
  test<"{:.6Lf}">(SV("0#000123"), F(1.234567e-4));
  test<"{:.6Lf}">(SV("0#001235"), F(1.234567e-3));
  test<"{:.6Lf}">(SV("0#012346"), F(1.234567e-2));
  test<"{:.6Lf}">(SV("0#123457"), F(1.234567e-1));
  test<"{:.6Lf}">(SV("1#234567"), F(1.234567e0));
  test<"{:.6Lf}">(SV("1_2#345670"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("12_3#456700"), F(1.234567e2));
    test<"{:.6Lf}">(SV("1_23_4#567000"), F(1.234567e3));
    test<"{:.6Lf}">(SV("12_34_5#670000"), F(1.234567e4));
    test<"{:.6Lf}">(SV("123_45_6#700000"), F(1.234567e5));
    test<"{:.6Lf}">(SV("1_234_56_7#000000"), F(1.234567e6));
    test<"{:.6Lf}">(SV("12_345_67_0#000000"), F(1.234567e7));
    test<"{:.6Lf}">(SV("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), F(1.234567e20));
  }
  test<"{:.6Lf}">(SV("-0#000001"), F(-1.234567e-6));
  test<"{:.6Lf}">(SV("-0#000012"), F(-1.234567e-5));
  test<"{:.6Lf}">(SV("-0#000123"), F(-1.234567e-4));
  test<"{:.6Lf}">(SV("-0#001235"), F(-1.234567e-3));
  test<"{:.6Lf}">(SV("-0#012346"), F(-1.234567e-2));
  test<"{:.6Lf}">(SV("-0#123457"), F(-1.234567e-1));
  test<"{:.6Lf}">(SV("-1#234567"), F(-1.234567e0));
  test<"{:.6Lf}">(SV("-1_2#345670"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("-12_3#456700"), F(-1.234567e2));
    test<"{:.6Lf}">(SV("-1_23_4#567000"), F(-1.234567e3));
    test<"{:.6Lf}">(SV("-12_34_5#670000"), F(-1.234567e4));
    test<"{:.6Lf}">(SV("-123_45_6#700000"), F(-1.234567e5));
    test<"{:.6Lf}">(SV("-1_234_56_7#000000"), F(-1.234567e6));
    test<"{:.6Lf}">(SV("-12_345_67_0#000000"), F(-1.234567e7));
    test<"{:.6Lf}">(SV("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), F(-1.234567e20));
  }

  test<"{:.6Lf}">(SV("0.000001"), en_US, F(1.234567e-6));
  test<"{:.6Lf}">(SV("0.000012"), en_US, F(1.234567e-5));
  test<"{:.6Lf}">(SV("0.000123"), en_US, F(1.234567e-4));
  test<"{:.6Lf}">(SV("0.001235"), en_US, F(1.234567e-3));
  test<"{:.6Lf}">(SV("0.012346"), en_US, F(1.234567e-2));
  test<"{:.6Lf}">(SV("0.123457"), en_US, F(1.234567e-1));
  test<"{:.6Lf}">(SV("1.234567"), en_US, F(1.234567e0));
  test<"{:.6Lf}">(SV("12.345670"), en_US, F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("123.456700"), en_US, F(1.234567e2));
    test<"{:.6Lf}">(SV("1,234.567000"), en_US, F(1.234567e3));
    test<"{:.6Lf}">(SV("12,345.670000"), en_US, F(1.234567e4));
    test<"{:.6Lf}">(SV("123,456.700000"), en_US, F(1.234567e5));
    test<"{:.6Lf}">(SV("1,234,567.000000"), en_US, F(1.234567e6));
    test<"{:.6Lf}">(SV("12,345,670.000000"), en_US, F(1.234567e7));
    test<"{:.6Lf}">(SV("123,456,700,000,000,000,000.000000"), en_US, F(1.234567e20));
  }
  test<"{:.6Lf}">(SV("-0.000001"), en_US, F(-1.234567e-6));
  test<"{:.6Lf}">(SV("-0.000012"), en_US, F(-1.234567e-5));
  test<"{:.6Lf}">(SV("-0.000123"), en_US, F(-1.234567e-4));
  test<"{:.6Lf}">(SV("-0.001235"), en_US, F(-1.234567e-3));
  test<"{:.6Lf}">(SV("-0.012346"), en_US, F(-1.234567e-2));
  test<"{:.6Lf}">(SV("-0.123457"), en_US, F(-1.234567e-1));
  test<"{:.6Lf}">(SV("-1.234567"), en_US, F(-1.234567e0));
  test<"{:.6Lf}">(SV("-12.345670"), en_US, F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("-123.456700"), en_US, F(-1.234567e2));
    test<"{:.6Lf}">(SV("-1,234.567000"), en_US, F(-1.234567e3));
    test<"{:.6Lf}">(SV("-12,345.670000"), en_US, F(-1.234567e4));
    test<"{:.6Lf}">(SV("-123,456.700000"), en_US, F(-1.234567e5));
    test<"{:.6Lf}">(SV("-1,234,567.000000"), en_US, F(-1.234567e6));
    test<"{:.6Lf}">(SV("-12,345,670.000000"), en_US, F(-1.234567e7));
    test<"{:.6Lf}">(SV("-123,456,700,000,000,000,000.000000"), en_US, F(-1.234567e20));
  }

  std::locale::global(en_US);
  test<"{:.6Lf}">(SV("0#000001"), loc, F(1.234567e-6));
  test<"{:.6Lf}">(SV("0#000012"), loc, F(1.234567e-5));
  test<"{:.6Lf}">(SV("0#000123"), loc, F(1.234567e-4));
  test<"{:.6Lf}">(SV("0#001235"), loc, F(1.234567e-3));
  test<"{:.6Lf}">(SV("0#012346"), loc, F(1.234567e-2));
  test<"{:.6Lf}">(SV("0#123457"), loc, F(1.234567e-1));
  test<"{:.6Lf}">(SV("1#234567"), loc, F(1.234567e0));
  test<"{:.6Lf}">(SV("1_2#345670"), loc, F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("12_3#456700"), loc, F(1.234567e2));
    test<"{:.6Lf}">(SV("1_23_4#567000"), loc, F(1.234567e3));
    test<"{:.6Lf}">(SV("12_34_5#670000"), loc, F(1.234567e4));
    test<"{:.6Lf}">(SV("123_45_6#700000"), loc, F(1.234567e5));
    test<"{:.6Lf}">(SV("1_234_56_7#000000"), loc, F(1.234567e6));
    test<"{:.6Lf}">(SV("12_345_67_0#000000"), loc, F(1.234567e7));
    test<"{:.6Lf}">(SV("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), loc, F(1.234567e20));
  }
  test<"{:.6Lf}">(SV("-0#000001"), loc, F(-1.234567e-6));
  test<"{:.6Lf}">(SV("-0#000012"), loc, F(-1.234567e-5));
  test<"{:.6Lf}">(SV("-0#000123"), loc, F(-1.234567e-4));
  test<"{:.6Lf}">(SV("-0#001235"), loc, F(-1.234567e-3));
  test<"{:.6Lf}">(SV("-0#012346"), loc, F(-1.234567e-2));
  test<"{:.6Lf}">(SV("-0#123457"), loc, F(-1.234567e-1));
  test<"{:.6Lf}">(SV("-1#234567"), loc, F(-1.234567e0));
  test<"{:.6Lf}">(SV("-1_2#345670"), loc, F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:.6Lf}">(SV("-12_3#456700"), loc, F(-1.234567e2));
    test<"{:.6Lf}">(SV("-1_23_4#567000"), loc, F(-1.234567e3));
    test<"{:.6Lf}">(SV("-12_34_5#670000"), loc, F(-1.234567e4));
    test<"{:.6Lf}">(SV("-123_45_6#700000"), loc, F(-1.234567e5));
    test<"{:.6Lf}">(SV("-1_234_56_7#000000"), loc, F(-1.234567e6));
    test<"{:.6Lf}">(SV("-12_345_67_0#000000"), loc, F(-1.234567e7));
    test<"{:.6Lf}">(SV("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), loc, F(-1.234567e20));
  }

  // *** Fill, align, zero padding ***
  if constexpr (sizeof(F) > sizeof(float)) {
    std::locale::global(en_US);
    test<"{:$<15.6Lf}">(SV("1,234.567000$$$"), F(1.234567e3));
    test<"{:$>15.6Lf}">(SV("$$$1,234.567000"), F(1.234567e3));
    test<"{:$^15.6Lf}">(SV("$1,234.567000$$"), F(1.234567e3));
    test<"{:015.6Lf}">(SV("0001,234.567000"), F(1.234567e3));
    test<"{:$<16.6Lf}">(SV("-1,234.567000$$$"), F(-1.234567e3));
    test<"{:$>16.6Lf}">(SV("$$$-1,234.567000"), F(-1.234567e3));
    test<"{:$^16.6Lf}">(SV("$-1,234.567000$$"), F(-1.234567e3));
    test<"{:016.6Lf}">(SV("-0001,234.567000"), F(-1.234567e3));

    std::locale::global(loc);
    test<"{:$<16.6Lf}">(SV("1_23_4#567000$$$"), F(1.234567e3));
    test<"{:$>16.6Lf}">(SV("$$$1_23_4#567000"), F(1.234567e3));
    test<"{:$^16.6Lf}">(SV("$1_23_4#567000$$"), F(1.234567e3));
    test<"{:016.6Lf}">(SV("0001_23_4#567000"), F(1.234567e3));
    test<"{:$<17.6Lf}">(SV("-1_23_4#567000$$$"), F(-1.234567e3));
    test<"{:$>17.6Lf}">(SV("$$$-1_23_4#567000"), F(-1.234567e3));
    test<"{:$^17.6Lf}">(SV("$-1_23_4#567000$$"), F(-1.234567e3));
    test<"{:017.6Lf}">(SV("-0001_23_4#567000"), F(-1.234567e3));

    test<"{:$<15.6Lf}">(SV("1,234.567000$$$"), en_US, F(1.234567e3));
    test<"{:$>15.6Lf}">(SV("$$$1,234.567000"), en_US, F(1.234567e3));
    test<"{:$^15.6Lf}">(SV("$1,234.567000$$"), en_US, F(1.234567e3));
    test<"{:015.6Lf}">(SV("0001,234.567000"), en_US, F(1.234567e3));
    test<"{:$<16.6Lf}">(SV("-1,234.567000$$$"), en_US, F(-1.234567e3));
    test<"{:$>16.6Lf}">(SV("$$$-1,234.567000"), en_US, F(-1.234567e3));
    test<"{:$^16.6Lf}">(SV("$-1,234.567000$$"), en_US, F(-1.234567e3));
    test<"{:016.6Lf}">(SV("-0001,234.567000"), en_US, F(-1.234567e3));

    std::locale::global(en_US);
    test<"{:$<16.6Lf}">(SV("1_23_4#567000$$$"), loc, F(1.234567e3));
    test<"{:$>16.6Lf}">(SV("$$$1_23_4#567000"), loc, F(1.234567e3));
    test<"{:$^16.6Lf}">(SV("$1_23_4#567000$$"), loc, F(1.234567e3));
    test<"{:016.6Lf}">(SV("0001_23_4#567000"), loc, F(1.234567e3));
    test<"{:$<17.6Lf}">(SV("-1_23_4#567000$$$"), loc, F(-1.234567e3));
    test<"{:$>17.6Lf}">(SV("$$$-1_23_4#567000"), loc, F(-1.234567e3));
    test<"{:$^17.6Lf}">(SV("$-1_23_4#567000$$"), loc, F(-1.234567e3));
    test<"{:017.6Lf}">(SV("-0001_23_4#567000"), loc, F(-1.234567e3));
  }
}

template <class F, class CharT>
void test_floating_point_general_lower_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test<"{:.6Lg}">(SV("1.23457e-06"), F(1.234567e-6));
  test<"{:.6Lg}">(SV("1.23457e-05"), F(1.234567e-5));
  test<"{:.6Lg}">(SV("0.000123457"), F(1.234567e-4));
  test<"{:.6Lg}">(SV("0.00123457"), F(1.234567e-3));
  test<"{:.6Lg}">(SV("0.0123457"), F(1.234567e-2));
  test<"{:.6Lg}">(SV("0.123457"), F(1.234567e-1));
  test<"{:.6Lg}">(SV("1.23457"), F(1.234567e0));
  test<"{:.6Lg}">(SV("12.3457"), F(1.234567e1));
  test<"{:.6Lg}">(SV("123.457"), F(1.234567e2));
  test<"{:.6Lg}">(SV("1,234.57"), F(1.234567e3));
  test<"{:.6Lg}">(SV("12,345.7"), F(1.234567e4));
  test<"{:.6Lg}">(SV("123,457"), F(1.234567e5));
  test<"{:.6Lg}">(SV("1.23457e+06"), F(1.234567e6));
  test<"{:.6Lg}">(SV("1.23457e+07"), F(1.234567e7));
  test<"{:.6Lg}">(SV("-1.23457e-06"), F(-1.234567e-6));
  test<"{:.6Lg}">(SV("-1.23457e-05"), F(-1.234567e-5));
  test<"{:.6Lg}">(SV("-0.000123457"), F(-1.234567e-4));
  test<"{:.6Lg}">(SV("-0.00123457"), F(-1.234567e-3));
  test<"{:.6Lg}">(SV("-0.0123457"), F(-1.234567e-2));
  test<"{:.6Lg}">(SV("-0.123457"), F(-1.234567e-1));
  test<"{:.6Lg}">(SV("-1.23457"), F(-1.234567e0));
  test<"{:.6Lg}">(SV("-12.3457"), F(-1.234567e1));
  test<"{:.6Lg}">(SV("-123.457"), F(-1.234567e2));
  test<"{:.6Lg}">(SV("-1,234.57"), F(-1.234567e3));
  test<"{:.6Lg}">(SV("-12,345.7"), F(-1.234567e4));
  test<"{:.6Lg}">(SV("-123,457"), F(-1.234567e5));
  test<"{:.6Lg}">(SV("-1.23457e+06"), F(-1.234567e6));
  test<"{:.6Lg}">(SV("-1.23457e+07"), F(-1.234567e7));

  std::locale::global(loc);
  test<"{:.6Lg}">(SV("1#23457e-06"), F(1.234567e-6));
  test<"{:.6Lg}">(SV("1#23457e-05"), F(1.234567e-5));
  test<"{:.6Lg}">(SV("0#000123457"), F(1.234567e-4));
  test<"{:.6Lg}">(SV("0#00123457"), F(1.234567e-3));
  test<"{:.6Lg}">(SV("0#0123457"), F(1.234567e-2));
  test<"{:.6Lg}">(SV("0#123457"), F(1.234567e-1));
  test<"{:.6Lg}">(SV("1#23457"), F(1.234567e0));
  test<"{:.6Lg}">(SV("1_2#3457"), F(1.234567e1));
  test<"{:.6Lg}">(SV("12_3#457"), F(1.234567e2));
  test<"{:.6Lg}">(SV("1_23_4#57"), F(1.234567e3));
  test<"{:.6Lg}">(SV("12_34_5#7"), F(1.234567e4));
  test<"{:.6Lg}">(SV("123_45_7"), F(1.234567e5));
  test<"{:.6Lg}">(SV("1#23457e+06"), F(1.234567e6));
  test<"{:.6Lg}">(SV("1#23457e+07"), F(1.234567e7));
  test<"{:.6Lg}">(SV("-1#23457e-06"), F(-1.234567e-6));
  test<"{:.6Lg}">(SV("-1#23457e-05"), F(-1.234567e-5));
  test<"{:.6Lg}">(SV("-0#000123457"), F(-1.234567e-4));
  test<"{:.6Lg}">(SV("-0#00123457"), F(-1.234567e-3));
  test<"{:.6Lg}">(SV("-0#0123457"), F(-1.234567e-2));
  test<"{:.6Lg}">(SV("-0#123457"), F(-1.234567e-1));
  test<"{:.6Lg}">(SV("-1#23457"), F(-1.234567e0));
  test<"{:.6Lg}">(SV("-1_2#3457"), F(-1.234567e1));
  test<"{:.6Lg}">(SV("-12_3#457"), F(-1.234567e2));
  test<"{:.6Lg}">(SV("-1_23_4#57"), F(-1.234567e3));
  test<"{:.6Lg}">(SV("-12_34_5#7"), F(-1.234567e4));
  test<"{:.6Lg}">(SV("-123_45_7"), F(-1.234567e5));
  test<"{:.6Lg}">(SV("-1#23457e+06"), F(-1.234567e6));
  test<"{:.6Lg}">(SV("-1#23457e+07"), F(-1.234567e7));

  test<"{:.6Lg}">(SV("1.23457e-06"), en_US, F(1.234567e-6));
  test<"{:.6Lg}">(SV("1.23457e-05"), en_US, F(1.234567e-5));
  test<"{:.6Lg}">(SV("0.000123457"), en_US, F(1.234567e-4));
  test<"{:.6Lg}">(SV("0.00123457"), en_US, F(1.234567e-3));
  test<"{:.6Lg}">(SV("0.0123457"), en_US, F(1.234567e-2));
  test<"{:.6Lg}">(SV("0.123457"), en_US, F(1.234567e-1));
  test<"{:.6Lg}">(SV("1.23457"), en_US, F(1.234567e0));
  test<"{:.6Lg}">(SV("12.3457"), en_US, F(1.234567e1));
  test<"{:.6Lg}">(SV("123.457"), en_US, F(1.234567e2));
  test<"{:.6Lg}">(SV("1,234.57"), en_US, F(1.234567e3));
  test<"{:.6Lg}">(SV("12,345.7"), en_US, F(1.234567e4));
  test<"{:.6Lg}">(SV("123,457"), en_US, F(1.234567e5));
  test<"{:.6Lg}">(SV("1.23457e+06"), en_US, F(1.234567e6));
  test<"{:.6Lg}">(SV("1.23457e+07"), en_US, F(1.234567e7));
  test<"{:.6Lg}">(SV("-1.23457e-06"), en_US, F(-1.234567e-6));
  test<"{:.6Lg}">(SV("-1.23457e-05"), en_US, F(-1.234567e-5));
  test<"{:.6Lg}">(SV("-0.000123457"), en_US, F(-1.234567e-4));
  test<"{:.6Lg}">(SV("-0.00123457"), en_US, F(-1.234567e-3));
  test<"{:.6Lg}">(SV("-0.0123457"), en_US, F(-1.234567e-2));
  test<"{:.6Lg}">(SV("-0.123457"), en_US, F(-1.234567e-1));
  test<"{:.6Lg}">(SV("-1.23457"), en_US, F(-1.234567e0));
  test<"{:.6Lg}">(SV("-12.3457"), en_US, F(-1.234567e1));
  test<"{:.6Lg}">(SV("-123.457"), en_US, F(-1.234567e2));
  test<"{:.6Lg}">(SV("-1,234.57"), en_US, F(-1.234567e3));
  test<"{:.6Lg}">(SV("-12,345.7"), en_US, F(-1.234567e4));
  test<"{:.6Lg}">(SV("-123,457"), en_US, F(-1.234567e5));
  test<"{:.6Lg}">(SV("-1.23457e+06"), en_US, F(-1.234567e6));
  test<"{:.6Lg}">(SV("-1.23457e+07"), en_US, F(-1.234567e7));

  std::locale::global(en_US);
  test<"{:.6Lg}">(SV("1#23457e-06"), loc, F(1.234567e-6));
  test<"{:.6Lg}">(SV("1#23457e-05"), loc, F(1.234567e-5));
  test<"{:.6Lg}">(SV("0#000123457"), loc, F(1.234567e-4));
  test<"{:.6Lg}">(SV("0#00123457"), loc, F(1.234567e-3));
  test<"{:.6Lg}">(SV("0#0123457"), loc, F(1.234567e-2));
  test<"{:.6Lg}">(SV("0#123457"), loc, F(1.234567e-1));
  test<"{:.6Lg}">(SV("1#23457"), loc, F(1.234567e0));
  test<"{:.6Lg}">(SV("1_2#3457"), loc, F(1.234567e1));
  test<"{:.6Lg}">(SV("12_3#457"), loc, F(1.234567e2));
  test<"{:.6Lg}">(SV("1_23_4#57"), loc, F(1.234567e3));
  test<"{:.6Lg}">(SV("12_34_5#7"), loc, F(1.234567e4));
  test<"{:.6Lg}">(SV("123_45_7"), loc, F(1.234567e5));
  test<"{:.6Lg}">(SV("1#23457e+06"), loc, F(1.234567e6));
  test<"{:.6Lg}">(SV("1#23457e+07"), loc, F(1.234567e7));
  test<"{:.6Lg}">(SV("-1#23457e-06"), loc, F(-1.234567e-6));
  test<"{:.6Lg}">(SV("-1#23457e-05"), loc, F(-1.234567e-5));
  test<"{:.6Lg}">(SV("-0#000123457"), loc, F(-1.234567e-4));
  test<"{:.6Lg}">(SV("-0#00123457"), loc, F(-1.234567e-3));
  test<"{:.6Lg}">(SV("-0#0123457"), loc, F(-1.234567e-2));
  test<"{:.6Lg}">(SV("-0#123457"), loc, F(-1.234567e-1));
  test<"{:.6Lg}">(SV("-1#23457"), loc, F(-1.234567e0));
  test<"{:.6Lg}">(SV("-1_2#3457"), loc, F(-1.234567e1));
  test<"{:.6Lg}">(SV("-12_3#457"), loc, F(-1.234567e2));
  test<"{:.6Lg}">(SV("-1_23_4#57"), loc, F(-1.234567e3));
  test<"{:.6Lg}">(SV("-12_34_5#7"), loc, F(-1.234567e4));
  test<"{:.6Lg}">(SV("-123_45_7"), loc, F(-1.234567e5));
  test<"{:.6Lg}">(SV("-1#23457e+06"), loc, F(-1.234567e6));
  test<"{:.6Lg}">(SV("-1#23457e+07"), loc, F(-1.234567e7));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test<"{:$<11.6Lg}">(SV("1,234.57$$$"), F(1.234567e3));
  test<"{:$>11.6Lg}">(SV("$$$1,234.57"), F(1.234567e3));
  test<"{:$^11.6Lg}">(SV("$1,234.57$$"), F(1.234567e3));
  test<"{:011.6Lg}">(SV("0001,234.57"), F(1.234567e3));
  test<"{:$<12.6Lg}">(SV("-1,234.57$$$"), F(-1.234567e3));
  test<"{:$>12.6Lg}">(SV("$$$-1,234.57"), F(-1.234567e3));
  test<"{:$^12.6Lg}">(SV("$-1,234.57$$"), F(-1.234567e3));
  test<"{:012.6Lg}">(SV("-0001,234.57"), F(-1.234567e3));

  std::locale::global(loc);
  test<"{:$<12.6Lg}">(SV("1_23_4#57$$$"), F(1.234567e3));
  test<"{:$>12.6Lg}">(SV("$$$1_23_4#57"), F(1.234567e3));
  test<"{:$^12.6Lg}">(SV("$1_23_4#57$$"), F(1.234567e3));
  test<"{:012.6Lg}">(SV("0001_23_4#57"), F(1.234567e3));
  test<"{:$<13.6Lg}">(SV("-1_23_4#57$$$"), F(-1.234567e3));
  test<"{:$>13.6Lg}">(SV("$$$-1_23_4#57"), F(-1.234567e3));
  test<"{:$^13.6Lg}">(SV("$-1_23_4#57$$"), F(-1.234567e3));
  test<"{:013.6Lg}">(SV("-0001_23_4#57"), F(-1.234567e3));

  test<"{:$<11.6Lg}">(SV("1,234.57$$$"), en_US, F(1.234567e3));
  test<"{:$>11.6Lg}">(SV("$$$1,234.57"), en_US, F(1.234567e3));
  test<"{:$^11.6Lg}">(SV("$1,234.57$$"), en_US, F(1.234567e3));
  test<"{:011.6Lg}">(SV("0001,234.57"), en_US, F(1.234567e3));
  test<"{:$<12.6Lg}">(SV("-1,234.57$$$"), en_US, F(-1.234567e3));
  test<"{:$>12.6Lg}">(SV("$$$-1,234.57"), en_US, F(-1.234567e3));
  test<"{:$^12.6Lg}">(SV("$-1,234.57$$"), en_US, F(-1.234567e3));
  test<"{:012.6Lg}">(SV("-0001,234.57"), en_US, F(-1.234567e3));

  std::locale::global(en_US);
  test<"{:$<12.6Lg}">(SV("1_23_4#57$$$"), loc, F(1.234567e3));
  test<"{:$>12.6Lg}">(SV("$$$1_23_4#57"), loc, F(1.234567e3));
  test<"{:$^12.6Lg}">(SV("$1_23_4#57$$"), loc, F(1.234567e3));
  test<"{:012.6Lg}">(SV("0001_23_4#57"), loc, F(1.234567e3));
  test<"{:$<13.6Lg}">(SV("-1_23_4#57$$$"), loc, F(-1.234567e3));
  test<"{:$>13.6Lg}">(SV("$$$-1_23_4#57"), loc, F(-1.234567e3));
  test<"{:$^13.6Lg}">(SV("$-1_23_4#57$$"), loc, F(-1.234567e3));
  test<"{:013.6Lg}">(SV("-0001_23_4#57"), loc, F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_general_upper_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test<"{:.6LG}">(SV("1.23457E-06"), F(1.234567e-6));
  test<"{:.6LG}">(SV("1.23457E-05"), F(1.234567e-5));
  test<"{:.6LG}">(SV("0.000123457"), F(1.234567e-4));
  test<"{:.6LG}">(SV("0.00123457"), F(1.234567e-3));
  test<"{:.6LG}">(SV("0.0123457"), F(1.234567e-2));
  test<"{:.6LG}">(SV("0.123457"), F(1.234567e-1));
  test<"{:.6LG}">(SV("1.23457"), F(1.234567e0));
  test<"{:.6LG}">(SV("12.3457"), F(1.234567e1));
  test<"{:.6LG}">(SV("123.457"), F(1.234567e2));
  test<"{:.6LG}">(SV("1,234.57"), F(1.234567e3));
  test<"{:.6LG}">(SV("12,345.7"), F(1.234567e4));
  test<"{:.6LG}">(SV("123,457"), F(1.234567e5));
  test<"{:.6LG}">(SV("1.23457E+06"), F(1.234567e6));
  test<"{:.6LG}">(SV("1.23457E+07"), F(1.234567e7));
  test<"{:.6LG}">(SV("-1.23457E-06"), F(-1.234567e-6));
  test<"{:.6LG}">(SV("-1.23457E-05"), F(-1.234567e-5));
  test<"{:.6LG}">(SV("-0.000123457"), F(-1.234567e-4));
  test<"{:.6LG}">(SV("-0.00123457"), F(-1.234567e-3));
  test<"{:.6LG}">(SV("-0.0123457"), F(-1.234567e-2));
  test<"{:.6LG}">(SV("-0.123457"), F(-1.234567e-1));
  test<"{:.6LG}">(SV("-1.23457"), F(-1.234567e0));
  test<"{:.6LG}">(SV("-12.3457"), F(-1.234567e1));
  test<"{:.6LG}">(SV("-123.457"), F(-1.234567e2));
  test<"{:.6LG}">(SV("-1,234.57"), F(-1.234567e3));
  test<"{:.6LG}">(SV("-12,345.7"), F(-1.234567e4));
  test<"{:.6LG}">(SV("-123,457"), F(-1.234567e5));
  test<"{:.6LG}">(SV("-1.23457E+06"), F(-1.234567e6));
  test<"{:.6LG}">(SV("-1.23457E+07"), F(-1.234567e7));

  std::locale::global(loc);
  test<"{:.6LG}">(SV("1#23457E-06"), F(1.234567e-6));
  test<"{:.6LG}">(SV("1#23457E-05"), F(1.234567e-5));
  test<"{:.6LG}">(SV("0#000123457"), F(1.234567e-4));
  test<"{:.6LG}">(SV("0#00123457"), F(1.234567e-3));
  test<"{:.6LG}">(SV("0#0123457"), F(1.234567e-2));
  test<"{:.6LG}">(SV("0#123457"), F(1.234567e-1));
  test<"{:.6LG}">(SV("1#23457"), F(1.234567e0));
  test<"{:.6LG}">(SV("1_2#3457"), F(1.234567e1));
  test<"{:.6LG}">(SV("12_3#457"), F(1.234567e2));
  test<"{:.6LG}">(SV("1_23_4#57"), F(1.234567e3));
  test<"{:.6LG}">(SV("12_34_5#7"), F(1.234567e4));
  test<"{:.6LG}">(SV("123_45_7"), F(1.234567e5));
  test<"{:.6LG}">(SV("1#23457E+06"), F(1.234567e6));
  test<"{:.6LG}">(SV("1#23457E+07"), F(1.234567e7));
  test<"{:.6LG}">(SV("-1#23457E-06"), F(-1.234567e-6));
  test<"{:.6LG}">(SV("-1#23457E-05"), F(-1.234567e-5));
  test<"{:.6LG}">(SV("-0#000123457"), F(-1.234567e-4));
  test<"{:.6LG}">(SV("-0#00123457"), F(-1.234567e-3));
  test<"{:.6LG}">(SV("-0#0123457"), F(-1.234567e-2));
  test<"{:.6LG}">(SV("-0#123457"), F(-1.234567e-1));
  test<"{:.6LG}">(SV("-1#23457"), F(-1.234567e0));
  test<"{:.6LG}">(SV("-1_2#3457"), F(-1.234567e1));
  test<"{:.6LG}">(SV("-12_3#457"), F(-1.234567e2));
  test<"{:.6LG}">(SV("-1_23_4#57"), F(-1.234567e3));
  test<"{:.6LG}">(SV("-12_34_5#7"), F(-1.234567e4));
  test<"{:.6LG}">(SV("-123_45_7"), F(-1.234567e5));
  test<"{:.6LG}">(SV("-1#23457E+06"), F(-1.234567e6));
  test<"{:.6LG}">(SV("-1#23457E+07"), F(-1.234567e7));

  test<"{:.6LG}">(SV("1.23457E-06"), en_US, F(1.234567e-6));
  test<"{:.6LG}">(SV("1.23457E-05"), en_US, F(1.234567e-5));
  test<"{:.6LG}">(SV("0.000123457"), en_US, F(1.234567e-4));
  test<"{:.6LG}">(SV("0.00123457"), en_US, F(1.234567e-3));
  test<"{:.6LG}">(SV("0.0123457"), en_US, F(1.234567e-2));
  test<"{:.6LG}">(SV("0.123457"), en_US, F(1.234567e-1));
  test<"{:.6LG}">(SV("1.23457"), en_US, F(1.234567e0));
  test<"{:.6LG}">(SV("12.3457"), en_US, F(1.234567e1));
  test<"{:.6LG}">(SV("123.457"), en_US, F(1.234567e2));
  test<"{:.6LG}">(SV("1,234.57"), en_US, F(1.234567e3));
  test<"{:.6LG}">(SV("12,345.7"), en_US, F(1.234567e4));
  test<"{:.6LG}">(SV("123,457"), en_US, F(1.234567e5));
  test<"{:.6LG}">(SV("1.23457E+06"), en_US, F(1.234567e6));
  test<"{:.6LG}">(SV("1.23457E+07"), en_US, F(1.234567e7));
  test<"{:.6LG}">(SV("-1.23457E-06"), en_US, F(-1.234567e-6));
  test<"{:.6LG}">(SV("-1.23457E-05"), en_US, F(-1.234567e-5));
  test<"{:.6LG}">(SV("-0.000123457"), en_US, F(-1.234567e-4));
  test<"{:.6LG}">(SV("-0.00123457"), en_US, F(-1.234567e-3));
  test<"{:.6LG}">(SV("-0.0123457"), en_US, F(-1.234567e-2));
  test<"{:.6LG}">(SV("-0.123457"), en_US, F(-1.234567e-1));
  test<"{:.6LG}">(SV("-1.23457"), en_US, F(-1.234567e0));
  test<"{:.6LG}">(SV("-12.3457"), en_US, F(-1.234567e1));
  test<"{:.6LG}">(SV("-123.457"), en_US, F(-1.234567e2));
  test<"{:.6LG}">(SV("-1,234.57"), en_US, F(-1.234567e3));
  test<"{:.6LG}">(SV("-12,345.7"), en_US, F(-1.234567e4));
  test<"{:.6LG}">(SV("-123,457"), en_US, F(-1.234567e5));
  test<"{:.6LG}">(SV("-1.23457E+06"), en_US, F(-1.234567e6));
  test<"{:.6LG}">(SV("-1.23457E+07"), en_US, F(-1.234567e7));

  std::locale::global(en_US);
  test<"{:.6LG}">(SV("1#23457E-06"), loc, F(1.234567e-6));
  test<"{:.6LG}">(SV("1#23457E-05"), loc, F(1.234567e-5));
  test<"{:.6LG}">(SV("0#000123457"), loc, F(1.234567e-4));
  test<"{:.6LG}">(SV("0#00123457"), loc, F(1.234567e-3));
  test<"{:.6LG}">(SV("0#0123457"), loc, F(1.234567e-2));
  test<"{:.6LG}">(SV("0#123457"), loc, F(1.234567e-1));
  test<"{:.6LG}">(SV("1#23457"), loc, F(1.234567e0));
  test<"{:.6LG}">(SV("1_2#3457"), loc, F(1.234567e1));
  test<"{:.6LG}">(SV("12_3#457"), loc, F(1.234567e2));
  test<"{:.6LG}">(SV("1_23_4#57"), loc, F(1.234567e3));
  test<"{:.6LG}">(SV("12_34_5#7"), loc, F(1.234567e4));
  test<"{:.6LG}">(SV("123_45_7"), loc, F(1.234567e5));
  test<"{:.6LG}">(SV("1#23457E+06"), loc, F(1.234567e6));
  test<"{:.6LG}">(SV("1#23457E+07"), loc, F(1.234567e7));
  test<"{:.6LG}">(SV("-1#23457E-06"), loc, F(-1.234567e-6));
  test<"{:.6LG}">(SV("-1#23457E-05"), loc, F(-1.234567e-5));
  test<"{:.6LG}">(SV("-0#000123457"), loc, F(-1.234567e-4));
  test<"{:.6LG}">(SV("-0#00123457"), loc, F(-1.234567e-3));
  test<"{:.6LG}">(SV("-0#0123457"), loc, F(-1.234567e-2));
  test<"{:.6LG}">(SV("-0#123457"), loc, F(-1.234567e-1));
  test<"{:.6LG}">(SV("-1#23457"), loc, F(-1.234567e0));
  test<"{:.6LG}">(SV("-1_2#3457"), loc, F(-1.234567e1));
  test<"{:.6LG}">(SV("-12_3#457"), loc, F(-1.234567e2));
  test<"{:.6LG}">(SV("-1_23_4#57"), loc, F(-1.234567e3));
  test<"{:.6LG}">(SV("-12_34_5#7"), loc, F(-1.234567e4));
  test<"{:.6LG}">(SV("-123_45_7"), loc, F(-1.234567e5));
  test<"{:.6LG}">(SV("-1#23457E+06"), loc, F(-1.234567e6));
  test<"{:.6LG}">(SV("-1#23457E+07"), loc, F(-1.234567e7));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test<"{:$<11.6LG}">(SV("1,234.57$$$"), F(1.234567e3));
  test<"{:$>11.6LG}">(SV("$$$1,234.57"), F(1.234567e3));
  test<"{:$^11.6LG}">(SV("$1,234.57$$"), F(1.234567e3));
  test<"{:011.6LG}">(SV("0001,234.57"), F(1.234567e3));
  test<"{:$<12.6LG}">(SV("-1,234.57$$$"), F(-1.234567e3));
  test<"{:$>12.6LG}">(SV("$$$-1,234.57"), F(-1.234567e3));
  test<"{:$^12.6LG}">(SV("$-1,234.57$$"), F(-1.234567e3));
  test<"{:012.6LG}">(SV("-0001,234.57"), F(-1.234567e3));

  std::locale::global(loc);
  test<"{:$<12.6LG}">(SV("1_23_4#57$$$"), F(1.234567e3));
  test<"{:$>12.6LG}">(SV("$$$1_23_4#57"), F(1.234567e3));
  test<"{:$^12.6LG}">(SV("$1_23_4#57$$"), F(1.234567e3));
  test<"{:012.6LG}">(SV("0001_23_4#57"), F(1.234567e3));
  test<"{:$<13.6LG}">(SV("-1_23_4#57$$$"), F(-1.234567e3));
  test<"{:$>13.6LG}">(SV("$$$-1_23_4#57"), F(-1.234567e3));
  test<"{:$^13.6LG}">(SV("$-1_23_4#57$$"), F(-1.234567e3));
  test<"{:013.6LG}">(SV("-0001_23_4#57"), F(-1.234567e3));

  test<"{:$<11.6LG}">(SV("1,234.57$$$"), en_US, F(1.234567e3));
  test<"{:$>11.6LG}">(SV("$$$1,234.57"), en_US, F(1.234567e3));
  test<"{:$^11.6LG}">(SV("$1,234.57$$"), en_US, F(1.234567e3));
  test<"{:011.6LG}">(SV("0001,234.57"), en_US, F(1.234567e3));
  test<"{:$<12.6LG}">(SV("-1,234.57$$$"), en_US, F(-1.234567e3));
  test<"{:$>12.6LG}">(SV("$$$-1,234.57"), en_US, F(-1.234567e3));
  test<"{:$^12.6LG}">(SV("$-1,234.57$$"), en_US, F(-1.234567e3));
  test<"{:012.6LG}">(SV("-0001,234.57"), en_US, F(-1.234567e3));

  std::locale::global(en_US);
  test<"{:$<12.6LG}">(SV("1_23_4#57$$$"), loc, F(1.234567e3));
  test<"{:$>12.6LG}">(SV("$$$1_23_4#57"), loc, F(1.234567e3));
  test<"{:$^12.6LG}">(SV("$1_23_4#57$$"), loc, F(1.234567e3));
  test<"{:012.6LG}">(SV("0001_23_4#57"), loc, F(1.234567e3));
  test<"{:$<13.6LG}">(SV("-1_23_4#57$$$"), loc, F(-1.234567e3));
  test<"{:$>13.6LG}">(SV("$$$-1_23_4#57"), loc, F(-1.234567e3));
  test<"{:$^13.6LG}">(SV("$-1_23_4#57$$"), loc, F(-1.234567e3));
  test<"{:013.6LG}">(SV("-0001_23_4#57"), loc, F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_default() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test<"{:L}">(SV("1.234567e-06"), F(1.234567e-6));
  test<"{:L}">(SV("1.234567e-05"), F(1.234567e-5));
  test<"{:L}">(SV("0.0001234567"), F(1.234567e-4));
  test<"{:L}">(SV("0.001234567"), F(1.234567e-3));
  test<"{:L}">(SV("0.01234567"), F(1.234567e-2));
  test<"{:L}">(SV("0.1234567"), F(1.234567e-1));
  test<"{:L}">(SV("1.234567"), F(1.234567e0));
  test<"{:L}">(SV("12.34567"), F(1.234567e1));
  test<"{:L}">(SV("123.4567"), F(1.234567e2));
  test<"{:L}">(SV("1,234.567"), F(1.234567e3));
  test<"{:L}">(SV("12,345.67"), F(1.234567e4));
  test<"{:L}">(SV("123,456.7"), F(1.234567e5));
  test<"{:L}">(SV("1,234,567"), F(1.234567e6));
  test<"{:L}">(SV("12,345,670"), F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:L}">(SV("123,456,700"), F(1.234567e8));
    test<"{:L}">(SV("1,234,567,000"), F(1.234567e9));
    test<"{:L}">(SV("12,345,670,000"), F(1.234567e10));
    test<"{:L}">(SV("123,456,700,000"), F(1.234567e11));
    test<"{:L}">(SV("1.234567e+12"), F(1.234567e12));
    test<"{:L}">(SV("1.234567e+13"), F(1.234567e13));
  }
  test<"{:L}">(SV("-1.234567e-06"), F(-1.234567e-6));
  test<"{:L}">(SV("-1.234567e-05"), F(-1.234567e-5));
  test<"{:L}">(SV("-0.0001234567"), F(-1.234567e-4));
  test<"{:L}">(SV("-0.001234567"), F(-1.234567e-3));
  test<"{:L}">(SV("-0.01234567"), F(-1.234567e-2));
  test<"{:L}">(SV("-0.1234567"), F(-1.234567e-1));
  test<"{:L}">(SV("-1.234567"), F(-1.234567e0));
  test<"{:L}">(SV("-12.34567"), F(-1.234567e1));
  test<"{:L}">(SV("-123.4567"), F(-1.234567e2));
  test<"{:L}">(SV("-1,234.567"), F(-1.234567e3));
  test<"{:L}">(SV("-12,345.67"), F(-1.234567e4));
  test<"{:L}">(SV("-123,456.7"), F(-1.234567e5));
  test<"{:L}">(SV("-1,234,567"), F(-1.234567e6));
  test<"{:L}">(SV("-12,345,670"), F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:L}">(SV("-123,456,700"), F(-1.234567e8));
    test<"{:L}">(SV("-1,234,567,000"), F(-1.234567e9));
    test<"{:L}">(SV("-12,345,670,000"), F(-1.234567e10));
    test<"{:L}">(SV("-123,456,700,000"), F(-1.234567e11));
    test<"{:L}">(SV("-1.234567e+12"), F(-1.234567e12));
    test<"{:L}">(SV("-1.234567e+13"), F(-1.234567e13));
  }

  std::locale::global(loc);
  test<"{:L}">(SV("1#234567e-06"), F(1.234567e-6));
  test<"{:L}">(SV("1#234567e-05"), F(1.234567e-5));
  test<"{:L}">(SV("0#0001234567"), F(1.234567e-4));
  test<"{:L}">(SV("0#001234567"), F(1.234567e-3));
  test<"{:L}">(SV("0#01234567"), F(1.234567e-2));
  test<"{:L}">(SV("0#1234567"), F(1.234567e-1));
  test<"{:L}">(SV("1#234567"), F(1.234567e0));
  test<"{:L}">(SV("1_2#34567"), F(1.234567e1));
  test<"{:L}">(SV("12_3#4567"), F(1.234567e2));
  test<"{:L}">(SV("1_23_4#567"), F(1.234567e3));
  test<"{:L}">(SV("12_34_5#67"), F(1.234567e4));
  test<"{:L}">(SV("123_45_6#7"), F(1.234567e5));
  test<"{:L}">(SV("1_234_56_7"), F(1.234567e6));
  test<"{:L}">(SV("12_345_67_0"), F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:L}">(SV("1_23_456_70_0"), F(1.234567e8));
    test<"{:L}">(SV("1_2_34_567_00_0"), F(1.234567e9));
    test<"{:L}">(SV("1_2_3_45_670_00_0"), F(1.234567e10));
    test<"{:L}">(SV("1_2_3_4_56_700_00_0"), F(1.234567e11));
    test<"{:L}">(SV("1#234567e+12"), F(1.234567e12));
    test<"{:L}">(SV("1#234567e+13"), F(1.234567e13));
  }
  test<"{:L}">(SV("-1#234567e-06"), F(-1.234567e-6));
  test<"{:L}">(SV("-1#234567e-05"), F(-1.234567e-5));
  test<"{:L}">(SV("-0#0001234567"), F(-1.234567e-4));
  test<"{:L}">(SV("-0#001234567"), F(-1.234567e-3));
  test<"{:L}">(SV("-0#01234567"), F(-1.234567e-2));
  test<"{:L}">(SV("-0#1234567"), F(-1.234567e-1));
  test<"{:L}">(SV("-1#234567"), F(-1.234567e0));
  test<"{:L}">(SV("-1_2#34567"), F(-1.234567e1));
  test<"{:L}">(SV("-12_3#4567"), F(-1.234567e2));
  test<"{:L}">(SV("-1_23_4#567"), F(-1.234567e3));
  test<"{:L}">(SV("-12_34_5#67"), F(-1.234567e4));
  test<"{:L}">(SV("-123_45_6#7"), F(-1.234567e5));
  test<"{:L}">(SV("-1_234_56_7"), F(-1.234567e6));
  test<"{:L}">(SV("-12_345_67_0"), F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:L}">(SV("-1_23_456_70_0"), F(-1.234567e8));
    test<"{:L}">(SV("-1_2_34_567_00_0"), F(-1.234567e9));
    test<"{:L}">(SV("-1_2_3_45_670_00_0"), F(-1.234567e10));
    test<"{:L}">(SV("-1_2_3_4_56_700_00_0"), F(-1.234567e11));
    test<"{:L}">(SV("-1#234567e+12"), F(-1.234567e12));
    test<"{:L}">(SV("-1#234567e+13"), F(-1.234567e13));
  }

  test<"{:L}">(SV("1.234567e-06"), en_US, F(1.234567e-6));
  test<"{:L}">(SV("1.234567e-05"), en_US, F(1.234567e-5));
  test<"{:L}">(SV("0.0001234567"), en_US, F(1.234567e-4));
  test<"{:L}">(SV("0.001234567"), en_US, F(1.234567e-3));
  test<"{:L}">(SV("0.01234567"), en_US, F(1.234567e-2));
  test<"{:L}">(SV("0.1234567"), en_US, F(1.234567e-1));
  test<"{:L}">(SV("1.234567"), en_US, F(1.234567e0));
  test<"{:L}">(SV("12.34567"), en_US, F(1.234567e1));
  test<"{:L}">(SV("123.4567"), en_US, F(1.234567e2));
  test<"{:L}">(SV("1,234.567"), en_US, F(1.234567e3));
  test<"{:L}">(SV("12,345.67"), en_US, F(1.234567e4));
  test<"{:L}">(SV("123,456.7"), en_US, F(1.234567e5));
  test<"{:L}">(SV("1,234,567"), en_US, F(1.234567e6));
  test<"{:L}">(SV("12,345,670"), en_US, F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:L}">(SV("123,456,700"), en_US, F(1.234567e8));
    test<"{:L}">(SV("1,234,567,000"), en_US, F(1.234567e9));
    test<"{:L}">(SV("12,345,670,000"), en_US, F(1.234567e10));
    test<"{:L}">(SV("123,456,700,000"), en_US, F(1.234567e11));
    test<"{:L}">(SV("1.234567e+12"), en_US, F(1.234567e12));
    test<"{:L}">(SV("1.234567e+13"), en_US, F(1.234567e13));
  }
  test<"{:L}">(SV("-1.234567e-06"), en_US, F(-1.234567e-6));
  test<"{:L}">(SV("-1.234567e-05"), en_US, F(-1.234567e-5));
  test<"{:L}">(SV("-0.0001234567"), en_US, F(-1.234567e-4));
  test<"{:L}">(SV("-0.001234567"), en_US, F(-1.234567e-3));
  test<"{:L}">(SV("-0.01234567"), en_US, F(-1.234567e-2));
  test<"{:L}">(SV("-0.1234567"), en_US, F(-1.234567e-1));
  test<"{:L}">(SV("-1.234567"), en_US, F(-1.234567e0));
  test<"{:L}">(SV("-12.34567"), en_US, F(-1.234567e1));
  test<"{:L}">(SV("-123.4567"), en_US, F(-1.234567e2));
  test<"{:L}">(SV("-1,234.567"), en_US, F(-1.234567e3));
  test<"{:L}">(SV("-12,345.67"), en_US, F(-1.234567e4));
  test<"{:L}">(SV("-123,456.7"), en_US, F(-1.234567e5));
  test<"{:L}">(SV("-1,234,567"), en_US, F(-1.234567e6));
  test<"{:L}">(SV("-12,345,670"), en_US, F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:L}">(SV("-123,456,700"), en_US, F(-1.234567e8));
    test<"{:L}">(SV("-1,234,567,000"), en_US, F(-1.234567e9));
    test<"{:L}">(SV("-12,345,670,000"), en_US, F(-1.234567e10));
    test<"{:L}">(SV("-123,456,700,000"), en_US, F(-1.234567e11));
    test<"{:L}">(SV("-1.234567e+12"), en_US, F(-1.234567e12));
    test<"{:L}">(SV("-1.234567e+13"), en_US, F(-1.234567e13));
  }

  std::locale::global(en_US);
  test<"{:L}">(SV("1#234567e-06"), loc, F(1.234567e-6));
  test<"{:L}">(SV("1#234567e-05"), loc, F(1.234567e-5));
  test<"{:L}">(SV("0#0001234567"), loc, F(1.234567e-4));
  test<"{:L}">(SV("0#001234567"), loc, F(1.234567e-3));
  test<"{:L}">(SV("0#01234567"), loc, F(1.234567e-2));
  test<"{:L}">(SV("0#1234567"), loc, F(1.234567e-1));
  test<"{:L}">(SV("1#234567"), loc, F(1.234567e0));
  test<"{:L}">(SV("1_2#34567"), loc, F(1.234567e1));
  test<"{:L}">(SV("12_3#4567"), loc, F(1.234567e2));
  test<"{:L}">(SV("1_23_4#567"), loc, F(1.234567e3));
  test<"{:L}">(SV("12_34_5#67"), loc, F(1.234567e4));
  test<"{:L}">(SV("123_45_6#7"), loc, F(1.234567e5));
  test<"{:L}">(SV("1_234_56_7"), loc, F(1.234567e6));
  test<"{:L}">(SV("12_345_67_0"), loc, F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:L}">(SV("1_23_456_70_0"), loc, F(1.234567e8));
    test<"{:L}">(SV("1_2_34_567_00_0"), loc, F(1.234567e9));
    test<"{:L}">(SV("1_2_3_45_670_00_0"), loc, F(1.234567e10));
    test<"{:L}">(SV("1_2_3_4_56_700_00_0"), loc, F(1.234567e11));
    test<"{:L}">(SV("1#234567e+12"), loc, F(1.234567e12));
    test<"{:L}">(SV("1#234567e+13"), loc, F(1.234567e13));
  }
  test<"{:L}">(SV("-1#234567e-06"), loc, F(-1.234567e-6));
  test<"{:L}">(SV("-1#234567e-05"), loc, F(-1.234567e-5));
  test<"{:L}">(SV("-0#0001234567"), loc, F(-1.234567e-4));
  test<"{:L}">(SV("-0#001234567"), loc, F(-1.234567e-3));
  test<"{:L}">(SV("-0#01234567"), loc, F(-1.234567e-2));
  test<"{:L}">(SV("-0#1234567"), loc, F(-1.234567e-1));
  test<"{:L}">(SV("-1#234567"), loc, F(-1.234567e0));
  test<"{:L}">(SV("-1_2#34567"), loc, F(-1.234567e1));
  test<"{:L}">(SV("-12_3#4567"), loc, F(-1.234567e2));
  test<"{:L}">(SV("-1_23_4#567"), loc, F(-1.234567e3));
  test<"{:L}">(SV("-12_34_5#67"), loc, F(-1.234567e4));
  test<"{:L}">(SV("-123_45_6#7"), loc, F(-1.234567e5));
  test<"{:L}">(SV("-1_234_56_7"), loc, F(-1.234567e6));
  test<"{:L}">(SV("-12_345_67_0"), loc, F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test<"{:L}">(SV("-1_23_456_70_0"), loc, F(-1.234567e8));
    test<"{:L}">(SV("-1_2_34_567_00_0"), loc, F(-1.234567e9));
    test<"{:L}">(SV("-1_2_3_45_670_00_0"), loc, F(-1.234567e10));
    test<"{:L}">(SV("-1_2_3_4_56_700_00_0"), loc, F(-1.234567e11));
    test<"{:L}">(SV("-1#234567e+12"), loc, F(-1.234567e12));
    test<"{:L}">(SV("-1#234567e+13"), loc, F(-1.234567e13));
  }

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test<"{:$<12L}">(SV("1,234.567$$$"), F(1.234567e3));
  test<"{:$>12L}">(SV("$$$1,234.567"), F(1.234567e3));
  test<"{:$^12L}">(SV("$1,234.567$$"), F(1.234567e3));
  test<"{:012L}">(SV("0001,234.567"), F(1.234567e3));
  test<"{:$<13L}">(SV("-1,234.567$$$"), F(-1.234567e3));
  test<"{:$>13L}">(SV("$$$-1,234.567"), F(-1.234567e3));
  test<"{:$^13L}">(SV("$-1,234.567$$"), F(-1.234567e3));
  test<"{:013L}">(SV("-0001,234.567"), F(-1.234567e3));

  std::locale::global(loc);
  test<"{:$<13L}">(SV("1_23_4#567$$$"), F(1.234567e3));
  test<"{:$>13L}">(SV("$$$1_23_4#567"), F(1.234567e3));
  test<"{:$^13L}">(SV("$1_23_4#567$$"), F(1.234567e3));
  test<"{:013L}">(SV("0001_23_4#567"), F(1.234567e3));
  test<"{:$<14L}">(SV("-1_23_4#567$$$"), F(-1.234567e3));
  test<"{:$>14L}">(SV("$$$-1_23_4#567"), F(-1.234567e3));
  test<"{:$^14L}">(SV("$-1_23_4#567$$"), F(-1.234567e3));
  test<"{:014L}">(SV("-0001_23_4#567"), F(-1.234567e3));

  test<"{:$<12L}">(SV("1,234.567$$$"), en_US, F(1.234567e3));
  test<"{:$>12L}">(SV("$$$1,234.567"), en_US, F(1.234567e3));
  test<"{:$^12L}">(SV("$1,234.567$$"), en_US, F(1.234567e3));
  test<"{:012L}">(SV("0001,234.567"), en_US, F(1.234567e3));
  test<"{:$<13L}">(SV("-1,234.567$$$"), en_US, F(-1.234567e3));
  test<"{:$>13L}">(SV("$$$-1,234.567"), en_US, F(-1.234567e3));
  test<"{:$^13L}">(SV("$-1,234.567$$"), en_US, F(-1.234567e3));
  test<"{:013L}">(SV("-0001,234.567"), en_US, F(-1.234567e3));

  std::locale::global(en_US);
  test<"{:$<13L}">(SV("1_23_4#567$$$"), loc, F(1.234567e3));
  test<"{:$>13L}">(SV("$$$1_23_4#567"), loc, F(1.234567e3));
  test<"{:$^13L}">(SV("$1_23_4#567$$"), loc, F(1.234567e3));
  test<"{:013L}">(SV("0001_23_4#567"), loc, F(1.234567e3));
  test<"{:$<14L}">(SV("-1_23_4#567$$$"), loc, F(-1.234567e3));
  test<"{:$>14L}">(SV("$$$-1_23_4#567"), loc, F(-1.234567e3));
  test<"{:$^14L}">(SV("$-1_23_4#567$$"), loc, F(-1.234567e3));
  test<"{:014L}">(SV("-0001_23_4#567"), loc, F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_default_precision() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test<"{:.6L}">(SV("1.23457e-06"), F(1.234567e-6));
  test<"{:.6L}">(SV("1.23457e-05"), F(1.234567e-5));
  test<"{:.6L}">(SV("0.000123457"), F(1.234567e-4));
  test<"{:.6L}">(SV("0.00123457"), F(1.234567e-3));
  test<"{:.6L}">(SV("0.0123457"), F(1.234567e-2));
  test<"{:.6L}">(SV("0.123457"), F(1.234567e-1));
  test<"{:.6L}">(SV("1.23457"), F(1.234567e0));
  test<"{:.6L}">(SV("12.3457"), F(1.234567e1));
  test<"{:.6L}">(SV("123.457"), F(1.234567e2));
  test<"{:.6L}">(SV("1,234.57"), F(1.234567e3));
  test<"{:.6L}">(SV("12,345.7"), F(1.234567e4));
  test<"{:.6L}">(SV("123,457"), F(1.234567e5));
  test<"{:.6L}">(SV("1.23457e+06"), F(1.234567e6));
  test<"{:.6L}">(SV("1.23457e+07"), F(1.234567e7));
  test<"{:.6L}">(SV("-1.23457e-06"), F(-1.234567e-6));
  test<"{:.6L}">(SV("-1.23457e-05"), F(-1.234567e-5));
  test<"{:.6L}">(SV("-0.000123457"), F(-1.234567e-4));
  test<"{:.6L}">(SV("-0.00123457"), F(-1.234567e-3));
  test<"{:.6L}">(SV("-0.0123457"), F(-1.234567e-2));
  test<"{:.6L}">(SV("-0.123457"), F(-1.234567e-1));
  test<"{:.6L}">(SV("-1.23457"), F(-1.234567e0));
  test<"{:.6L}">(SV("-12.3457"), F(-1.234567e1));
  test<"{:.6L}">(SV("-123.457"), F(-1.234567e2));
  test<"{:.6L}">(SV("-1,234.57"), F(-1.234567e3));
  test<"{:.6L}">(SV("-12,345.7"), F(-1.234567e4));
  test<"{:.6L}">(SV("-123,457"), F(-1.234567e5));
  test<"{:.6L}">(SV("-1.23457e+06"), F(-1.234567e6));
  test<"{:.6L}">(SV("-1.23457e+07"), F(-1.234567e7));

  std::locale::global(loc);
  test<"{:.6L}">(SV("1#23457e-06"), F(1.234567e-6));
  test<"{:.6L}">(SV("1#23457e-05"), F(1.234567e-5));
  test<"{:.6L}">(SV("0#000123457"), F(1.234567e-4));
  test<"{:.6L}">(SV("0#00123457"), F(1.234567e-3));
  test<"{:.6L}">(SV("0#0123457"), F(1.234567e-2));
  test<"{:.6L}">(SV("0#123457"), F(1.234567e-1));
  test<"{:.6L}">(SV("1#23457"), F(1.234567e0));
  test<"{:.6L}">(SV("1_2#3457"), F(1.234567e1));
  test<"{:.6L}">(SV("12_3#457"), F(1.234567e2));
  test<"{:.6L}">(SV("1_23_4#57"), F(1.234567e3));
  test<"{:.6L}">(SV("12_34_5#7"), F(1.234567e4));
  test<"{:.6L}">(SV("123_45_7"), F(1.234567e5));
  test<"{:.6L}">(SV("1#23457e+06"), F(1.234567e6));
  test<"{:.6L}">(SV("1#23457e+07"), F(1.234567e7));
  test<"{:.6L}">(SV("-1#23457e-06"), F(-1.234567e-6));
  test<"{:.6L}">(SV("-1#23457e-05"), F(-1.234567e-5));
  test<"{:.6L}">(SV("-0#000123457"), F(-1.234567e-4));
  test<"{:.6L}">(SV("-0#00123457"), F(-1.234567e-3));
  test<"{:.6L}">(SV("-0#0123457"), F(-1.234567e-2));
  test<"{:.6L}">(SV("-0#123457"), F(-1.234567e-1));
  test<"{:.6L}">(SV("-1#23457"), F(-1.234567e0));
  test<"{:.6L}">(SV("-1_2#3457"), F(-1.234567e1));
  test<"{:.6L}">(SV("-12_3#457"), F(-1.234567e2));
  test<"{:.6L}">(SV("-1_23_4#57"), F(-1.234567e3));
  test<"{:.6L}">(SV("-12_34_5#7"), F(-1.234567e4));
  test<"{:.6L}">(SV("-123_45_7"), F(-1.234567e5));
  test<"{:.6L}">(SV("-1#23457e+06"), F(-1.234567e6));
  test<"{:.6L}">(SV("-1#23457e+07"), F(-1.234567e7));

  test<"{:.6L}">(SV("1.23457e-06"), en_US, F(1.234567e-6));
  test<"{:.6L}">(SV("1.23457e-05"), en_US, F(1.234567e-5));
  test<"{:.6L}">(SV("0.000123457"), en_US, F(1.234567e-4));
  test<"{:.6L}">(SV("0.00123457"), en_US, F(1.234567e-3));
  test<"{:.6L}">(SV("0.0123457"), en_US, F(1.234567e-2));
  test<"{:.6L}">(SV("0.123457"), en_US, F(1.234567e-1));
  test<"{:.6L}">(SV("1.23457"), en_US, F(1.234567e0));
  test<"{:.6L}">(SV("12.3457"), en_US, F(1.234567e1));
  test<"{:.6L}">(SV("123.457"), en_US, F(1.234567e2));
  test<"{:.6L}">(SV("1,234.57"), en_US, F(1.234567e3));
  test<"{:.6L}">(SV("12,345.7"), en_US, F(1.234567e4));
  test<"{:.6L}">(SV("123,457"), en_US, F(1.234567e5));
  test<"{:.6L}">(SV("1.23457e+06"), en_US, F(1.234567e6));
  test<"{:.6L}">(SV("1.23457e+07"), en_US, F(1.234567e7));
  test<"{:.6L}">(SV("-1.23457e-06"), en_US, F(-1.234567e-6));
  test<"{:.6L}">(SV("-1.23457e-05"), en_US, F(-1.234567e-5));
  test<"{:.6L}">(SV("-0.000123457"), en_US, F(-1.234567e-4));
  test<"{:.6L}">(SV("-0.00123457"), en_US, F(-1.234567e-3));
  test<"{:.6L}">(SV("-0.0123457"), en_US, F(-1.234567e-2));
  test<"{:.6L}">(SV("-0.123457"), en_US, F(-1.234567e-1));
  test<"{:.6L}">(SV("-1.23457"), en_US, F(-1.234567e0));
  test<"{:.6L}">(SV("-12.3457"), en_US, F(-1.234567e1));
  test<"{:.6L}">(SV("-123.457"), en_US, F(-1.234567e2));
  test<"{:.6L}">(SV("-1,234.57"), en_US, F(-1.234567e3));
  test<"{:.6L}">(SV("-12,345.7"), en_US, F(-1.234567e4));
  test<"{:.6L}">(SV("-123,457"), en_US, F(-1.234567e5));
  test<"{:.6L}">(SV("-1.23457e+06"), en_US, F(-1.234567e6));
  test<"{:.6L}">(SV("-1.23457e+07"), en_US, F(-1.234567e7));

  std::locale::global(en_US);
  test<"{:.6L}">(SV("1#23457e-06"), loc, F(1.234567e-6));
  test<"{:.6L}">(SV("1#23457e-05"), loc, F(1.234567e-5));
  test<"{:.6L}">(SV("0#000123457"), loc, F(1.234567e-4));
  test<"{:.6L}">(SV("0#00123457"), loc, F(1.234567e-3));
  test<"{:.6L}">(SV("0#0123457"), loc, F(1.234567e-2));
  test<"{:.6L}">(SV("0#123457"), loc, F(1.234567e-1));
  test<"{:.6L}">(SV("1#23457"), loc, F(1.234567e0));
  test<"{:.6L}">(SV("1_2#3457"), loc, F(1.234567e1));
  test<"{:.6L}">(SV("12_3#457"), loc, F(1.234567e2));
  test<"{:.6L}">(SV("1_23_4#57"), loc, F(1.234567e3));
  test<"{:.6L}">(SV("12_34_5#7"), loc, F(1.234567e4));
  test<"{:.6L}">(SV("123_45_7"), loc, F(1.234567e5));
  test<"{:.6L}">(SV("1#23457e+06"), loc, F(1.234567e6));
  test<"{:.6L}">(SV("1#23457e+07"), loc, F(1.234567e7));
  test<"{:.6L}">(SV("-1#23457e-06"), loc, F(-1.234567e-6));
  test<"{:.6L}">(SV("-1#23457e-05"), loc, F(-1.234567e-5));
  test<"{:.6L}">(SV("-0#000123457"), loc, F(-1.234567e-4));
  test<"{:.6L}">(SV("-0#00123457"), loc, F(-1.234567e-3));
  test<"{:.6L}">(SV("-0#0123457"), loc, F(-1.234567e-2));
  test<"{:.6L}">(SV("-0#123457"), loc, F(-1.234567e-1));
  test<"{:.6L}">(SV("-1#23457"), loc, F(-1.234567e0));
  test<"{:.6L}">(SV("-1_2#3457"), loc, F(-1.234567e1));
  test<"{:.6L}">(SV("-12_3#457"), loc, F(-1.234567e2));
  test<"{:.6L}">(SV("-1_23_4#57"), loc, F(-1.234567e3));
  test<"{:.6L}">(SV("-12_34_5#7"), loc, F(-1.234567e4));
  test<"{:.6L}">(SV("-123_45_7"), loc, F(-1.234567e5));
  test<"{:.6L}">(SV("-1#23457e+06"), loc, F(-1.234567e6));
  test<"{:.6L}">(SV("-1#23457e+07"), loc, F(-1.234567e7));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test<"{:$<11.6L}">(SV("1,234.57$$$"), F(1.234567e3));
  test<"{:$>11.6L}">(SV("$$$1,234.57"), F(1.234567e3));
  test<"{:$^11.6L}">(SV("$1,234.57$$"), F(1.234567e3));
  test<"{:011.6L}">(SV("0001,234.57"), F(1.234567e3));
  test<"{:$<12.6L}">(SV("-1,234.57$$$"), F(-1.234567e3));
  test<"{:$>12.6L}">(SV("$$$-1,234.57"), F(-1.234567e3));
  test<"{:$^12.6L}">(SV("$-1,234.57$$"), F(-1.234567e3));
  test<"{:012.6L}">(SV("-0001,234.57"), F(-1.234567e3));

  std::locale::global(loc);
  test<"{:$<12.6L}">(SV("1_23_4#57$$$"), F(1.234567e3));
  test<"{:$>12.6L}">(SV("$$$1_23_4#57"), F(1.234567e3));
  test<"{:$^12.6L}">(SV("$1_23_4#57$$"), F(1.234567e3));
  test<"{:012.6L}">(SV("0001_23_4#57"), F(1.234567e3));
  test<"{:$<13.6L}">(SV("-1_23_4#57$$$"), F(-1.234567e3));
  test<"{:$>13.6L}">(SV("$$$-1_23_4#57"), F(-1.234567e3));
  test<"{:$^13.6L}">(SV("$-1_23_4#57$$"), F(-1.234567e3));
  test<"{:013.6L}">(SV("-0001_23_4#57"), F(-1.234567e3));

  test<"{:$<11.6L}">(SV("1,234.57$$$"), en_US, F(1.234567e3));
  test<"{:$>11.6L}">(SV("$$$1,234.57"), en_US, F(1.234567e3));
  test<"{:$^11.6L}">(SV("$1,234.57$$"), en_US, F(1.234567e3));
  test<"{:011.6L}">(SV("0001,234.57"), en_US, F(1.234567e3));
  test<"{:$<12.6L}">(SV("-1,234.57$$$"), en_US, F(-1.234567e3));
  test<"{:$>12.6L}">(SV("$$$-1,234.57"), en_US, F(-1.234567e3));
  test<"{:$^12.6L}">(SV("$-1,234.57$$"), en_US, F(-1.234567e3));
  test<"{:012.6L}">(SV("-0001,234.57"), en_US, F(-1.234567e3));

  std::locale::global(en_US);
  test<"{:$<12.6L}">(SV("1_23_4#57$$$"), loc, F(1.234567e3));
  test<"{:$>12.6L}">(SV("$$$1_23_4#57"), loc, F(1.234567e3));
  test<"{:$^12.6L}">(SV("$1_23_4#57$$"), loc, F(1.234567e3));
  test<"{:012.6L}">(SV("0001_23_4#57"), loc, F(1.234567e3));
  test<"{:$<13.6L}">(SV("-1_23_4#57$$$"), loc, F(-1.234567e3));
  test<"{:$>13.6L}">(SV("$$$-1_23_4#57"), loc, F(-1.234567e3));
  test<"{:$^13.6L}">(SV("$-1_23_4#57$$"), loc, F(-1.234567e3));
  test<"{:013.6L}">(SV("-0001_23_4#57"), loc, F(-1.234567e3));
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
