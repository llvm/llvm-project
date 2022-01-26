//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format

// The issue is caused in __format_spec::__determine_grouping().
// There a string iterator is modified. The string is returned
// from the dylib's use_facet<numpunct<_CharT>>::grouping()
// XFAIL: LIBCXX-DEBUG-FIXME

// TODO FMT Evaluate gcc-11 status
// UNSUPPORTED: gcc-11

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

#define STR(S) MAKE_STRING(CharT, S)

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
void test(std::basic_string<CharT> expected, std::basic_string<CharT> fmt, const Args&... args) {
  // *** format ***
  {
    std::basic_string<CharT> out = std::format(fmt, args...);
    if constexpr (std::same_as<CharT, char>)
      if (out != expected)
        std::cerr << "\nFormat string   " << fmt << "\nExpected output " << expected << "\nActual output   " << out
                  << '\n';
    assert(out == expected);
  }
  // *** vformat ***
  {
    std::basic_string<CharT> out = std::vformat(fmt, std::make_format_args<context_t<CharT>>(args...));
    assert(out == expected);
  }
  // *** format_to ***
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::format_to(out.begin(), fmt, args...);
    assert(it == out.end());
    assert(out == expected);
  }
  // *** vformat_to ***
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::vformat_to(out.begin(), fmt, std::make_format_args<context_t<CharT>>(args...));
    assert(it == out.end());
    assert(out == expected);
  }
  // *** format_to_n ***
  {
    std::basic_string<CharT> out;
    std::format_to_n_result result = std::format_to_n(std::back_inserter(out), 1000, fmt, args...);
    using diff_type = decltype(result.size);
    diff_type formatted_size = std::formatted_size(fmt, args...);
    diff_type size = std::min<diff_type>(1000, formatted_size);

    assert(result.size == formatted_size);
    assert(out == expected.substr(0, size));
  }
  // *** formatted_size ***
  {
    size_t size = std::formatted_size(fmt, args...);
    assert(size == expected.size());
  }
}

template <class CharT, class... Args>
void test(std::basic_string<CharT> expected, std::locale loc, std::basic_string<CharT> fmt, const Args&... args) {
  // *** format ***
  {
    std::basic_string<CharT> out = std::format(loc, fmt, args...);
    if constexpr (std::same_as<CharT, char>)
      if (out != expected)
        std::cerr << "\nFormat string   " << fmt << "\nExpected output " << expected << "\nActual output   " << out
                  << '\n';
    assert(out == expected);
  }
  // *** vformat ***
  {
    std::basic_string<CharT> out = std::vformat(loc, fmt, std::make_format_args<context_t<CharT>>(args...));
    assert(out == expected);
  }
  // *** format_to ***
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::format_to(out.begin(), loc, fmt, args...);
    assert(it == out.end());
    assert(out == expected);
  }
  // *** vformat_to ***
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::vformat_to(out.begin(), loc, fmt, std::make_format_args<context_t<CharT>>(args...));
    assert(it == out.end());
    assert(out == expected);
  }
  // *** format_to_n ***
  {
    std::basic_string<CharT> out;
    std::format_to_n_result result = std::format_to_n(std::back_inserter(out), 1000, loc, fmt, args...);
    using diff_type = decltype(result.size);
    diff_type formatted_size = std::formatted_size(loc, fmt, args...);
    diff_type size = std::min<diff_type>(1000, formatted_size);

    assert(result.size == formatted_size);
    assert(out == expected.substr(0, size));
  }
  // *** formatted_size ***
  {
    size_t size = std::formatted_size(loc, fmt, args...);
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
  test(STR("true"), STR("{:L}"), true);
  test(STR("false"), STR("{:L}"), false);

  test(STR("yes"), loc, STR("{:L}"), true);
  test(STR("no"), loc, STR("{:L}"), false);

  std::locale::global(loc);
  test(STR("yes"), STR("{:L}"), true);
  test(STR("no"), STR("{:L}"), false);

  test(STR("true"), std::locale(LOCALE_en_US_UTF_8), STR("{:L}"), true);
  test(STR("false"), std::locale(LOCALE_en_US_UTF_8), STR("{:L}"), false);

#ifndef TEST_HAS_NO_UNICODE
  std::locale loc_unicode = std::locale(std::locale(), new numpunct_unicode<CharT>());

  test(STR("gültig"), loc_unicode, STR("{:L}"), true);
  test(STR("ungültig"), loc_unicode, STR("{:L}"), false);

  test(STR("gültig   "), loc_unicode, STR("{:9L}"), true);
  test(STR("gültig!!!"), loc_unicode, STR("{:!<9L}"), true);
  test(STR("_gültig__"), loc_unicode, STR("{:_^9L}"), true);
  test(STR("   gültig"), loc_unicode, STR("{:>9L}"), true);
#endif // TEST_HAS_NO_UNICODE
}

template <class CharT>
void test_integer() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Decimal ***
  std::locale::global(en_US);
  test(STR("0"), STR("{:L}"), 0);
  test(STR("1"), STR("{:L}"), 1);
  test(STR("10"), STR("{:L}"), 10);
  test(STR("100"), STR("{:L}"), 100);
  test(STR("1,000"), STR("{:L}"), 1'000);
  test(STR("10,000"), STR("{:L}"), 10'000);
  test(STR("100,000"), STR("{:L}"), 100'000);
  test(STR("1,000,000"), STR("{:L}"), 1'000'000);
  test(STR("10,000,000"), STR("{:L}"), 10'000'000);
  test(STR("100,000,000"), STR("{:L}"), 100'000'000);
  test(STR("1,000,000,000"), STR("{:L}"), 1'000'000'000);

  test(STR("-1"), STR("{:L}"), -1);
  test(STR("-10"), STR("{:L}"), -10);
  test(STR("-100"), STR("{:L}"), -100);
  test(STR("-1,000"), STR("{:L}"), -1'000);
  test(STR("-10,000"), STR("{:L}"), -10'000);
  test(STR("-100,000"), STR("{:L}"), -100'000);
  test(STR("-1,000,000"), STR("{:L}"), -1'000'000);
  test(STR("-10,000,000"), STR("{:L}"), -10'000'000);
  test(STR("-100,000,000"), STR("{:L}"), -100'000'000);
  test(STR("-1,000,000,000"), STR("{:L}"), -1'000'000'000);

  std::locale::global(loc);
  test(STR("0"), STR("{:L}"), 0);
  test(STR("1"), STR("{:L}"), 1);
  test(STR("1_0"), STR("{:L}"), 10);
  test(STR("10_0"), STR("{:L}"), 100);
  test(STR("1_00_0"), STR("{:L}"), 1'000);
  test(STR("10_00_0"), STR("{:L}"), 10'000);
  test(STR("100_00_0"), STR("{:L}"), 100'000);
  test(STR("1_000_00_0"), STR("{:L}"), 1'000'000);
  test(STR("10_000_00_0"), STR("{:L}"), 10'000'000);
  test(STR("1_00_000_00_0"), STR("{:L}"), 100'000'000);
  test(STR("1_0_00_000_00_0"), STR("{:L}"), 1'000'000'000);

  test(STR("-1"), STR("{:L}"), -1);
  test(STR("-1_0"), STR("{:L}"), -10);
  test(STR("-10_0"), STR("{:L}"), -100);
  test(STR("-1_00_0"), STR("{:L}"), -1'000);
  test(STR("-10_00_0"), STR("{:L}"), -10'000);
  test(STR("-100_00_0"), STR("{:L}"), -100'000);
  test(STR("-1_000_00_0"), STR("{:L}"), -1'000'000);
  test(STR("-10_000_00_0"), STR("{:L}"), -10'000'000);
  test(STR("-1_00_000_00_0"), STR("{:L}"), -100'000'000);
  test(STR("-1_0_00_000_00_0"), STR("{:L}"), -1'000'000'000);

  test(STR("0"), en_US, STR("{:L}"), 0);
  test(STR("1"), en_US, STR("{:L}"), 1);
  test(STR("10"), en_US, STR("{:L}"), 10);
  test(STR("100"), en_US, STR("{:L}"), 100);
  test(STR("1,000"), en_US, STR("{:L}"), 1'000);
  test(STR("10,000"), en_US, STR("{:L}"), 10'000);
  test(STR("100,000"), en_US, STR("{:L}"), 100'000);
  test(STR("1,000,000"), en_US, STR("{:L}"), 1'000'000);
  test(STR("10,000,000"), en_US, STR("{:L}"), 10'000'000);
  test(STR("100,000,000"), en_US, STR("{:L}"), 100'000'000);
  test(STR("1,000,000,000"), en_US, STR("{:L}"), 1'000'000'000);

  test(STR("-1"), en_US, STR("{:L}"), -1);
  test(STR("-10"), en_US, STR("{:L}"), -10);
  test(STR("-100"), en_US, STR("{:L}"), -100);
  test(STR("-1,000"), en_US, STR("{:L}"), -1'000);
  test(STR("-10,000"), en_US, STR("{:L}"), -10'000);
  test(STR("-100,000"), en_US, STR("{:L}"), -100'000);
  test(STR("-1,000,000"), en_US, STR("{:L}"), -1'000'000);
  test(STR("-10,000,000"), en_US, STR("{:L}"), -10'000'000);
  test(STR("-100,000,000"), en_US, STR("{:L}"), -100'000'000);
  test(STR("-1,000,000,000"), en_US, STR("{:L}"), -1'000'000'000);

  std::locale::global(en_US);
  test(STR("0"), loc, STR("{:L}"), 0);
  test(STR("1"), loc, STR("{:L}"), 1);
  test(STR("1_0"), loc, STR("{:L}"), 10);
  test(STR("10_0"), loc, STR("{:L}"), 100);
  test(STR("1_00_0"), loc, STR("{:L}"), 1'000);
  test(STR("10_00_0"), loc, STR("{:L}"), 10'000);
  test(STR("100_00_0"), loc, STR("{:L}"), 100'000);
  test(STR("1_000_00_0"), loc, STR("{:L}"), 1'000'000);
  test(STR("10_000_00_0"), loc, STR("{:L}"), 10'000'000);
  test(STR("1_00_000_00_0"), loc, STR("{:L}"), 100'000'000);
  test(STR("1_0_00_000_00_0"), loc, STR("{:L}"), 1'000'000'000);

  test(STR("-1"), loc, STR("{:L}"), -1);
  test(STR("-1_0"), loc, STR("{:L}"), -10);
  test(STR("-10_0"), loc, STR("{:L}"), -100);
  test(STR("-1_00_0"), loc, STR("{:L}"), -1'000);
  test(STR("-10_00_0"), loc, STR("{:L}"), -10'000);
  test(STR("-100_00_0"), loc, STR("{:L}"), -100'000);
  test(STR("-1_000_00_0"), loc, STR("{:L}"), -1'000'000);
  test(STR("-10_000_00_0"), loc, STR("{:L}"), -10'000'000);
  test(STR("-1_00_000_00_0"), loc, STR("{:L}"), -100'000'000);
  test(STR("-1_0_00_000_00_0"), loc, STR("{:L}"), -1'000'000'000);

  // *** Binary ***
  std::locale::global(en_US);
  test(STR("0"), STR("{:Lb}"), 0b0);
  test(STR("1"), STR("{:Lb}"), 0b1);
  test(STR("1,000,000,000"), STR("{:Lb}"), 0b1'000'000'000);

  test(STR("0b0"), STR("{:#Lb}"), 0b0);
  test(STR("0b1"), STR("{:#Lb}"), 0b1);
  test(STR("0b1,000,000,000"), STR("{:#Lb}"), 0b1'000'000'000);

  test(STR("-1"), STR("{:LB}"), -0b1);
  test(STR("-1,000,000,000"), STR("{:LB}"), -0b1'000'000'000);

  test(STR("-0B1"), STR("{:#LB}"), -0b1);
  test(STR("-0B1,000,000,000"), STR("{:#LB}"), -0b1'000'000'000);

  std::locale::global(loc);
  test(STR("0"), STR("{:Lb}"), 0b0);
  test(STR("1"), STR("{:Lb}"), 0b1);
  test(STR("1_0_00_000_00_0"), STR("{:Lb}"), 0b1'000'000'000);

  test(STR("0b0"), STR("{:#Lb}"), 0b0);
  test(STR("0b1"), STR("{:#Lb}"), 0b1);
  test(STR("0b1_0_00_000_00_0"), STR("{:#Lb}"), 0b1'000'000'000);

  test(STR("-1"), STR("{:LB}"), -0b1);
  test(STR("-1_0_00_000_00_0"), STR("{:LB}"), -0b1'000'000'000);

  test(STR("-0B1"), STR("{:#LB}"), -0b1);
  test(STR("-0B1_0_00_000_00_0"), STR("{:#LB}"), -0b1'000'000'000);

  test(STR("0"), en_US, STR("{:Lb}"), 0b0);
  test(STR("1"), en_US, STR("{:Lb}"), 0b1);
  test(STR("1,000,000,000"), en_US, STR("{:Lb}"), 0b1'000'000'000);

  test(STR("0b0"), en_US, STR("{:#Lb}"), 0b0);
  test(STR("0b1"), en_US, STR("{:#Lb}"), 0b1);
  test(STR("0b1,000,000,000"), en_US, STR("{:#Lb}"), 0b1'000'000'000);

  test(STR("-1"), en_US, STR("{:LB}"), -0b1);
  test(STR("-1,000,000,000"), en_US, STR("{:LB}"), -0b1'000'000'000);

  test(STR("-0B1"), en_US, STR("{:#LB}"), -0b1);
  test(STR("-0B1,000,000,000"), en_US, STR("{:#LB}"), -0b1'000'000'000);

  std::locale::global(en_US);
  test(STR("0"), loc, STR("{:Lb}"), 0b0);
  test(STR("1"), loc, STR("{:Lb}"), 0b1);
  test(STR("1_0_00_000_00_0"), loc, STR("{:Lb}"), 0b1'000'000'000);

  test(STR("0b0"), loc, STR("{:#Lb}"), 0b0);
  test(STR("0b1"), loc, STR("{:#Lb}"), 0b1);
  test(STR("0b1_0_00_000_00_0"), loc, STR("{:#Lb}"), 0b1'000'000'000);

  test(STR("-1"), loc, STR("{:LB}"), -0b1);
  test(STR("-1_0_00_000_00_0"), loc, STR("{:LB}"), -0b1'000'000'000);

  test(STR("-0B1"), loc, STR("{:#LB}"), -0b1);
  test(STR("-0B1_0_00_000_00_0"), loc, STR("{:#LB}"), -0b1'000'000'000);

  // *** Octal ***
  std::locale::global(en_US);
  test(STR("0"), STR("{:Lo}"), 00);
  test(STR("1"), STR("{:Lo}"), 01);
  test(STR("1,000,000,000"), STR("{:Lo}"), 01'000'000'000);

  test(STR("0"), STR("{:#Lo}"), 00);
  test(STR("01"), STR("{:#Lo}"), 01);
  test(STR("01,000,000,000"), STR("{:#Lo}"), 01'000'000'000);

  test(STR("-1"), STR("{:Lo}"), -01);
  test(STR("-1,000,000,000"), STR("{:Lo}"), -01'000'000'000);

  test(STR("-01"), STR("{:#Lo}"), -01);
  test(STR("-01,000,000,000"), STR("{:#Lo}"), -01'000'000'000);

  std::locale::global(loc);
  test(STR("0"), STR("{:Lo}"), 00);
  test(STR("1"), STR("{:Lo}"), 01);
  test(STR("1_0_00_000_00_0"), STR("{:Lo}"), 01'000'000'000);

  test(STR("0"), STR("{:#Lo}"), 00);
  test(STR("01"), STR("{:#Lo}"), 01);
  test(STR("01_0_00_000_00_0"), STR("{:#Lo}"), 01'000'000'000);

  test(STR("-1"), STR("{:Lo}"), -01);
  test(STR("-1_0_00_000_00_0"), STR("{:Lo}"), -01'000'000'000);

  test(STR("-01"), STR("{:#Lo}"), -01);
  test(STR("-01_0_00_000_00_0"), STR("{:#Lo}"), -01'000'000'000);

  test(STR("0"), en_US, STR("{:Lo}"), 00);
  test(STR("1"), en_US, STR("{:Lo}"), 01);
  test(STR("1,000,000,000"), en_US, STR("{:Lo}"), 01'000'000'000);

  test(STR("0"), en_US, STR("{:#Lo}"), 00);
  test(STR("01"), en_US, STR("{:#Lo}"), 01);
  test(STR("01,000,000,000"), en_US, STR("{:#Lo}"), 01'000'000'000);

  test(STR("-1"), en_US, STR("{:Lo}"), -01);
  test(STR("-1,000,000,000"), en_US, STR("{:Lo}"), -01'000'000'000);

  test(STR("-01"), en_US, STR("{:#Lo}"), -01);
  test(STR("-01,000,000,000"), en_US, STR("{:#Lo}"), -01'000'000'000);

  std::locale::global(en_US);
  test(STR("0"), loc, STR("{:Lo}"), 00);
  test(STR("1"), loc, STR("{:Lo}"), 01);
  test(STR("1_0_00_000_00_0"), loc, STR("{:Lo}"), 01'000'000'000);

  test(STR("0"), loc, STR("{:#Lo}"), 00);
  test(STR("01"), loc, STR("{:#Lo}"), 01);
  test(STR("01_0_00_000_00_0"), loc, STR("{:#Lo}"), 01'000'000'000);

  test(STR("-1"), loc, STR("{:Lo}"), -01);
  test(STR("-1_0_00_000_00_0"), loc, STR("{:Lo}"), -01'000'000'000);

  test(STR("-01"), loc, STR("{:#Lo}"), -01);
  test(STR("-01_0_00_000_00_0"), loc, STR("{:#Lo}"), -01'000'000'000);

  // *** Hexadecimal ***
  std::locale::global(en_US);
  test(STR("0"), STR("{:Lx}"), 0x0);
  test(STR("1"), STR("{:Lx}"), 0x1);
  test(STR("1,000,000,000"), STR("{:Lx}"), 0x1'000'000'000);

  test(STR("0x0"), STR("{:#Lx}"), 0x0);
  test(STR("0x1"), STR("{:#Lx}"), 0x1);
  test(STR("0x1,000,000,000"), STR("{:#Lx}"), 0x1'000'000'000);

  test(STR("-1"), STR("{:LX}"), -0x1);
  test(STR("-1,000,000,000"), STR("{:LX}"), -0x1'000'000'000);

  test(STR("-0X1"), STR("{:#LX}"), -0x1);
  test(STR("-0X1,000,000,000"), STR("{:#LX}"), -0x1'000'000'000);

  std::locale::global(loc);
  test(STR("0"), STR("{:Lx}"), 0x0);
  test(STR("1"), STR("{:Lx}"), 0x1);
  test(STR("1_0_00_000_00_0"), STR("{:Lx}"), 0x1'000'000'000);

  test(STR("0x0"), STR("{:#Lx}"), 0x0);
  test(STR("0x1"), STR("{:#Lx}"), 0x1);
  test(STR("0x1_0_00_000_00_0"), STR("{:#Lx}"), 0x1'000'000'000);

  test(STR("-1"), STR("{:LX}"), -0x1);
  test(STR("-1_0_00_000_00_0"), STR("{:LX}"), -0x1'000'000'000);

  test(STR("-0X1"), STR("{:#LX}"), -0x1);
  test(STR("-0X1_0_00_000_00_0"), STR("{:#LX}"), -0x1'000'000'000);

  test(STR("0"), en_US, STR("{:Lx}"), 0x0);
  test(STR("1"), en_US, STR("{:Lx}"), 0x1);
  test(STR("1,000,000,000"), en_US, STR("{:Lx}"), 0x1'000'000'000);

  test(STR("0x0"), en_US, STR("{:#Lx}"), 0x0);
  test(STR("0x1"), en_US, STR("{:#Lx}"), 0x1);
  test(STR("0x1,000,000,000"), en_US, STR("{:#Lx}"), 0x1'000'000'000);

  test(STR("-1"), en_US, STR("{:LX}"), -0x1);
  test(STR("-1,000,000,000"), en_US, STR("{:LX}"), -0x1'000'000'000);

  test(STR("-0X1"), en_US, STR("{:#LX}"), -0x1);
  test(STR("-0X1,000,000,000"), en_US, STR("{:#LX}"), -0x1'000'000'000);

  std::locale::global(en_US);
  test(STR("0"), loc, STR("{:Lx}"), 0x0);
  test(STR("1"), loc, STR("{:Lx}"), 0x1);
  test(STR("1_0_00_000_00_0"), loc, STR("{:Lx}"), 0x1'000'000'000);

  test(STR("0x0"), loc, STR("{:#Lx}"), 0x0);
  test(STR("0x1"), loc, STR("{:#Lx}"), 0x1);
  test(STR("0x1_0_00_000_00_0"), loc, STR("{:#Lx}"), 0x1'000'000'000);

  test(STR("-1"), loc, STR("{:LX}"), -0x1);
  test(STR("-1_0_00_000_00_0"), loc, STR("{:LX}"), -0x1'000'000'000);

  test(STR("-0X1"), loc, STR("{:#LX}"), -0x1);
  test(STR("-0X1_0_00_000_00_0"), loc, STR("{:#LX}"), -0x1'000'000'000);

  // *** align-fill & width ***
  test(STR("4_2"), loc, STR("{:L}"), 42);

  test(STR("   4_2"), loc, STR("{:6L}"), 42);
  test(STR("4_2   "), loc, STR("{:<6L}"), 42);
  test(STR(" 4_2  "), loc, STR("{:^6L}"), 42);
  test(STR("   4_2"), loc, STR("{:>6L}"), 42);

  test(STR("4_2***"), loc, STR("{:*<6L}"), 42);
  test(STR("*4_2**"), loc, STR("{:*^6L}"), 42);
  test(STR("***4_2"), loc, STR("{:*>6L}"), 42);

  test(STR("4_a*****"), loc, STR("{:*<8Lx}"), 0x4a);
  test(STR("**4_a***"), loc, STR("{:*^8Lx}"), 0x4a);
  test(STR("*****4_a"), loc, STR("{:*>8Lx}"), 0x4a);

  test(STR("0x4_a***"), loc, STR("{:*<#8Lx}"), 0x4a);
  test(STR("*0x4_a**"), loc, STR("{:*^#8Lx}"), 0x4a);
  test(STR("***0x4_a"), loc, STR("{:*>#8Lx}"), 0x4a);

  test(STR("4_A*****"), loc, STR("{:*<8LX}"), 0x4a);
  test(STR("**4_A***"), loc, STR("{:*^8LX}"), 0x4a);
  test(STR("*****4_A"), loc, STR("{:*>8LX}"), 0x4a);

  test(STR("0X4_A***"), loc, STR("{:*<#8LX}"), 0x4a);
  test(STR("*0X4_A**"), loc, STR("{:*^#8LX}"), 0x4a);
  test(STR("***0X4_A"), loc, STR("{:*>#8LX}"), 0x4a);

  // Test whether zero padding is ignored
  test(STR("4_2   "), loc, STR("{:<06L}"), 42);
  test(STR(" 4_2  "), loc, STR("{:^06L}"), 42);
  test(STR("   4_2"), loc, STR("{:>06L}"), 42);

  // *** zero-padding & width ***
  test(STR("   4_2"), loc, STR("{:6L}"), 42);
  test(STR("0004_2"), loc, STR("{:06L}"), 42);
  test(STR("-004_2"), loc, STR("{:06L}"), -42);

  test(STR("000004_a"), loc, STR("{:08Lx}"), 0x4a);
  test(STR("0x0004_a"), loc, STR("{:#08Lx}"), 0x4a);
  test(STR("0X0004_A"), loc, STR("{:#08LX}"), 0x4a);

  test(STR("-00004_a"), loc, STR("{:08Lx}"), -0x4a);
  test(STR("-0x004_a"), loc, STR("{:#08Lx}"), -0x4a);
  test(STR("-0X004_A"), loc, STR("{:#08LX}"), -0x4a);
}

template <class F, class CharT>
void test_floating_point_hex_lower_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(STR("1.23456p-3"), STR("{:La}"), F(0x1.23456p-3));
  test(STR("1.23456p-2"), STR("{:La}"), F(0x1.23456p-2));
  test(STR("1.23456p-1"), STR("{:La}"), F(0x1.23456p-1));
  test(STR("1.23456p+0"), STR("{:La}"), F(0x1.23456p0));
  test(STR("1.23456p+1"), STR("{:La}"), F(0x1.23456p+1));
  test(STR("1.23456p+2"), STR("{:La}"), F(0x1.23456p+2));
  test(STR("1.23456p+3"), STR("{:La}"), F(0x1.23456p+3));
  test(STR("1.23456p+20"), STR("{:La}"), F(0x1.23456p+20));

  std::locale::global(loc);
  test(STR("1#23456p-3"), STR("{:La}"), F(0x1.23456p-3));
  test(STR("1#23456p-2"), STR("{:La}"), F(0x1.23456p-2));
  test(STR("1#23456p-1"), STR("{:La}"), F(0x1.23456p-1));
  test(STR("1#23456p+0"), STR("{:La}"), F(0x1.23456p0));
  test(STR("1#23456p+1"), STR("{:La}"), F(0x1.23456p+1));
  test(STR("1#23456p+2"), STR("{:La}"), F(0x1.23456p+2));
  test(STR("1#23456p+3"), STR("{:La}"), F(0x1.23456p+3));
  test(STR("1#23456p+20"), STR("{:La}"), F(0x1.23456p+20));

  test(STR("1.23456p-3"), en_US, STR("{:La}"), F(0x1.23456p-3));
  test(STR("1.23456p-2"), en_US, STR("{:La}"), F(0x1.23456p-2));
  test(STR("1.23456p-1"), en_US, STR("{:La}"), F(0x1.23456p-1));
  test(STR("1.23456p+0"), en_US, STR("{:La}"), F(0x1.23456p0));
  test(STR("1.23456p+1"), en_US, STR("{:La}"), F(0x1.23456p+1));
  test(STR("1.23456p+2"), en_US, STR("{:La}"), F(0x1.23456p+2));
  test(STR("1.23456p+3"), en_US, STR("{:La}"), F(0x1.23456p+3));
  test(STR("1.23456p+20"), en_US, STR("{:La}"), F(0x1.23456p+20));

  std::locale::global(en_US);
  test(STR("1#23456p-3"), loc, STR("{:La}"), F(0x1.23456p-3));
  test(STR("1#23456p-2"), loc, STR("{:La}"), F(0x1.23456p-2));
  test(STR("1#23456p-1"), loc, STR("{:La}"), F(0x1.23456p-1));
  test(STR("1#23456p+0"), loc, STR("{:La}"), F(0x1.23456p0));
  test(STR("1#23456p+1"), loc, STR("{:La}"), F(0x1.23456p+1));
  test(STR("1#23456p+2"), loc, STR("{:La}"), F(0x1.23456p+2));
  test(STR("1#23456p+3"), loc, STR("{:La}"), F(0x1.23456p+3));
  test(STR("1#23456p+20"), loc, STR("{:La}"), F(0x1.23456p+20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(STR("1.23456p+3$$$"), STR("{:$<13La}"), F(0x1.23456p3));
  test(STR("$$$1.23456p+3"), STR("{:$>13La}"), F(0x1.23456p3));
  test(STR("$1.23456p+3$$"), STR("{:$^13La}"), F(0x1.23456p3));
  test(STR("0001.23456p+3"), STR("{:013La}"), F(0x1.23456p3));
  test(STR("-1.23456p+3$$$"), STR("{:$<14La}"), F(-0x1.23456p3));
  test(STR("$$$-1.23456p+3"), STR("{:$>14La}"), F(-0x1.23456p3));
  test(STR("$-1.23456p+3$$"), STR("{:$^14La}"), F(-0x1.23456p3));
  test(STR("-0001.23456p+3"), STR("{:014La}"), F(-0x1.23456p3));

  std::locale::global(loc);
  test(STR("1#23456p+3$$$"), STR("{:$<13La}"), F(0x1.23456p3));
  test(STR("$$$1#23456p+3"), STR("{:$>13La}"), F(0x1.23456p3));
  test(STR("$1#23456p+3$$"), STR("{:$^13La}"), F(0x1.23456p3));
  test(STR("0001#23456p+3"), STR("{:013La}"), F(0x1.23456p3));
  test(STR("-1#23456p+3$$$"), STR("{:$<14La}"), F(-0x1.23456p3));
  test(STR("$$$-1#23456p+3"), STR("{:$>14La}"), F(-0x1.23456p3));
  test(STR("$-1#23456p+3$$"), STR("{:$^14La}"), F(-0x1.23456p3));
  test(STR("-0001#23456p+3"), STR("{:014La}"), F(-0x1.23456p3));

  test(STR("1.23456p+3$$$"), en_US, STR("{:$<13La}"), F(0x1.23456p3));
  test(STR("$$$1.23456p+3"), en_US, STR("{:$>13La}"), F(0x1.23456p3));
  test(STR("$1.23456p+3$$"), en_US, STR("{:$^13La}"), F(0x1.23456p3));
  test(STR("0001.23456p+3"), en_US, STR("{:013La}"), F(0x1.23456p3));
  test(STR("-1.23456p+3$$$"), en_US, STR("{:$<14La}"), F(-0x1.23456p3));
  test(STR("$$$-1.23456p+3"), en_US, STR("{:$>14La}"), F(-0x1.23456p3));
  test(STR("$-1.23456p+3$$"), en_US, STR("{:$^14La}"), F(-0x1.23456p3));
  test(STR("-0001.23456p+3"), en_US, STR("{:014La}"), F(-0x1.23456p3));

  std::locale::global(en_US);
  test(STR("1#23456p+3$$$"), loc, STR("{:$<13La}"), F(0x1.23456p3));
  test(STR("$$$1#23456p+3"), loc, STR("{:$>13La}"), F(0x1.23456p3));
  test(STR("$1#23456p+3$$"), loc, STR("{:$^13La}"), F(0x1.23456p3));
  test(STR("0001#23456p+3"), loc, STR("{:013La}"), F(0x1.23456p3));
  test(STR("-1#23456p+3$$$"), loc, STR("{:$<14La}"), F(-0x1.23456p3));
  test(STR("$$$-1#23456p+3"), loc, STR("{:$>14La}"), F(-0x1.23456p3));
  test(STR("$-1#23456p+3$$"), loc, STR("{:$^14La}"), F(-0x1.23456p3));
  test(STR("-0001#23456p+3"), loc, STR("{:014La}"), F(-0x1.23456p3));
}

template <class F, class CharT>
void test_floating_point_hex_upper_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(STR("1.23456P-3"), STR("{:LA}"), F(0x1.23456p-3));
  test(STR("1.23456P-2"), STR("{:LA}"), F(0x1.23456p-2));
  test(STR("1.23456P-1"), STR("{:LA}"), F(0x1.23456p-1));
  test(STR("1.23456P+0"), STR("{:LA}"), F(0x1.23456p0));
  test(STR("1.23456P+1"), STR("{:LA}"), F(0x1.23456p+1));
  test(STR("1.23456P+2"), STR("{:LA}"), F(0x1.23456p+2));
  test(STR("1.23456P+3"), STR("{:LA}"), F(0x1.23456p+3));
  test(STR("1.23456P+20"), STR("{:LA}"), F(0x1.23456p+20));

  std::locale::global(loc);
  test(STR("1#23456P-3"), STR("{:LA}"), F(0x1.23456p-3));
  test(STR("1#23456P-2"), STR("{:LA}"), F(0x1.23456p-2));
  test(STR("1#23456P-1"), STR("{:LA}"), F(0x1.23456p-1));
  test(STR("1#23456P+0"), STR("{:LA}"), F(0x1.23456p0));
  test(STR("1#23456P+1"), STR("{:LA}"), F(0x1.23456p+1));
  test(STR("1#23456P+2"), STR("{:LA}"), F(0x1.23456p+2));
  test(STR("1#23456P+3"), STR("{:LA}"), F(0x1.23456p+3));
  test(STR("1#23456P+20"), STR("{:LA}"), F(0x1.23456p+20));

  test(STR("1.23456P-3"), en_US, STR("{:LA}"), F(0x1.23456p-3));
  test(STR("1.23456P-2"), en_US, STR("{:LA}"), F(0x1.23456p-2));
  test(STR("1.23456P-1"), en_US, STR("{:LA}"), F(0x1.23456p-1));
  test(STR("1.23456P+0"), en_US, STR("{:LA}"), F(0x1.23456p0));
  test(STR("1.23456P+1"), en_US, STR("{:LA}"), F(0x1.23456p+1));
  test(STR("1.23456P+2"), en_US, STR("{:LA}"), F(0x1.23456p+2));
  test(STR("1.23456P+3"), en_US, STR("{:LA}"), F(0x1.23456p+3));
  test(STR("1.23456P+20"), en_US, STR("{:LA}"), F(0x1.23456p+20));

  std::locale::global(en_US);
  test(STR("1#23456P-3"), loc, STR("{:LA}"), F(0x1.23456p-3));
  test(STR("1#23456P-2"), loc, STR("{:LA}"), F(0x1.23456p-2));
  test(STR("1#23456P-1"), loc, STR("{:LA}"), F(0x1.23456p-1));
  test(STR("1#23456P+0"), loc, STR("{:LA}"), F(0x1.23456p0));
  test(STR("1#23456P+1"), loc, STR("{:LA}"), F(0x1.23456p+1));
  test(STR("1#23456P+2"), loc, STR("{:LA}"), F(0x1.23456p+2));
  test(STR("1#23456P+3"), loc, STR("{:LA}"), F(0x1.23456p+3));
  test(STR("1#23456P+20"), loc, STR("{:LA}"), F(0x1.23456p+20));

  // *** Fill, align, zero Padding ***
  std::locale::global(en_US);
  test(STR("1.23456P+3$$$"), STR("{:$<13LA}"), F(0x1.23456p3));
  test(STR("$$$1.23456P+3"), STR("{:$>13LA}"), F(0x1.23456p3));
  test(STR("$1.23456P+3$$"), STR("{:$^13LA}"), F(0x1.23456p3));
  test(STR("0001.23456P+3"), STR("{:013LA}"), F(0x1.23456p3));
  test(STR("-1.23456P+3$$$"), STR("{:$<14LA}"), F(-0x1.23456p3));
  test(STR("$$$-1.23456P+3"), STR("{:$>14LA}"), F(-0x1.23456p3));
  test(STR("$-1.23456P+3$$"), STR("{:$^14LA}"), F(-0x1.23456p3));
  test(STR("-0001.23456P+3"), STR("{:014LA}"), F(-0x1.23456p3));

  std::locale::global(loc);
  test(STR("1#23456P+3$$$"), STR("{:$<13LA}"), F(0x1.23456p3));
  test(STR("$$$1#23456P+3"), STR("{:$>13LA}"), F(0x1.23456p3));
  test(STR("$1#23456P+3$$"), STR("{:$^13LA}"), F(0x1.23456p3));
  test(STR("0001#23456P+3"), STR("{:013LA}"), F(0x1.23456p3));
  test(STR("-1#23456P+3$$$"), STR("{:$<14LA}"), F(-0x1.23456p3));
  test(STR("$$$-1#23456P+3"), STR("{:$>14LA}"), F(-0x1.23456p3));
  test(STR("$-1#23456P+3$$"), STR("{:$^14LA}"), F(-0x1.23456p3));
  test(STR("-0001#23456P+3"), STR("{:014LA}"), F(-0x1.23456p3));

  test(STR("1.23456P+3$$$"), en_US, STR("{:$<13LA}"), F(0x1.23456p3));
  test(STR("$$$1.23456P+3"), en_US, STR("{:$>13LA}"), F(0x1.23456p3));
  test(STR("$1.23456P+3$$"), en_US, STR("{:$^13LA}"), F(0x1.23456p3));
  test(STR("0001.23456P+3"), en_US, STR("{:013LA}"), F(0x1.23456p3));
  test(STR("-1.23456P+3$$$"), en_US, STR("{:$<14LA}"), F(-0x1.23456p3));
  test(STR("$$$-1.23456P+3"), en_US, STR("{:$>14LA}"), F(-0x1.23456p3));
  test(STR("$-1.23456P+3$$"), en_US, STR("{:$^14LA}"), F(-0x1.23456p3));
  test(STR("-0001.23456P+3"), en_US, STR("{:014LA}"), F(-0x1.23456p3));

  std::locale::global(en_US);
  test(STR("1#23456P+3$$$"), loc, STR("{:$<13LA}"), F(0x1.23456p3));
  test(STR("$$$1#23456P+3"), loc, STR("{:$>13LA}"), F(0x1.23456p3));
  test(STR("$1#23456P+3$$"), loc, STR("{:$^13LA}"), F(0x1.23456p3));
  test(STR("0001#23456P+3"), loc, STR("{:013LA}"), F(0x1.23456p3));
  test(STR("-1#23456P+3$$$"), loc, STR("{:$<14LA}"), F(-0x1.23456p3));
  test(STR("$$$-1#23456P+3"), loc, STR("{:$>14LA}"), F(-0x1.23456p3));
  test(STR("$-1#23456P+3$$"), loc, STR("{:$^14LA}"), F(-0x1.23456p3));
  test(STR("-0001#23456P+3"), loc, STR("{:014LA}"), F(-0x1.23456p3));
}

template <class F, class CharT>
void test_floating_point_hex_lower_case_precision() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(STR("1.234560p-3"), STR("{:.6La}"), F(0x1.23456p-3));
  test(STR("1.234560p-2"), STR("{:.6La}"), F(0x1.23456p-2));
  test(STR("1.234560p-1"), STR("{:.6La}"), F(0x1.23456p-1));
  test(STR("1.234560p+0"), STR("{:.6La}"), F(0x1.23456p0));
  test(STR("1.234560p+1"), STR("{:.6La}"), F(0x1.23456p+1));
  test(STR("1.234560p+2"), STR("{:.6La}"), F(0x1.23456p+2));
  test(STR("1.234560p+3"), STR("{:.6La}"), F(0x1.23456p+3));
  test(STR("1.234560p+20"), STR("{:.6La}"), F(0x1.23456p+20));

  std::locale::global(loc);
  test(STR("1#234560p-3"), STR("{:.6La}"), F(0x1.23456p-3));
  test(STR("1#234560p-2"), STR("{:.6La}"), F(0x1.23456p-2));
  test(STR("1#234560p-1"), STR("{:.6La}"), F(0x1.23456p-1));
  test(STR("1#234560p+0"), STR("{:.6La}"), F(0x1.23456p0));
  test(STR("1#234560p+1"), STR("{:.6La}"), F(0x1.23456p+1));
  test(STR("1#234560p+2"), STR("{:.6La}"), F(0x1.23456p+2));
  test(STR("1#234560p+3"), STR("{:.6La}"), F(0x1.23456p+3));
  test(STR("1#234560p+20"), STR("{:.6La}"), F(0x1.23456p+20));

  test(STR("1.234560p-3"), en_US, STR("{:.6La}"), F(0x1.23456p-3));
  test(STR("1.234560p-2"), en_US, STR("{:.6La}"), F(0x1.23456p-2));
  test(STR("1.234560p-1"), en_US, STR("{:.6La}"), F(0x1.23456p-1));
  test(STR("1.234560p+0"), en_US, STR("{:.6La}"), F(0x1.23456p0));
  test(STR("1.234560p+1"), en_US, STR("{:.6La}"), F(0x1.23456p+1));
  test(STR("1.234560p+2"), en_US, STR("{:.6La}"), F(0x1.23456p+2));
  test(STR("1.234560p+3"), en_US, STR("{:.6La}"), F(0x1.23456p+3));
  test(STR("1.234560p+20"), en_US, STR("{:.6La}"), F(0x1.23456p+20));

  std::locale::global(en_US);
  test(STR("1#234560p-3"), loc, STR("{:.6La}"), F(0x1.23456p-3));
  test(STR("1#234560p-2"), loc, STR("{:.6La}"), F(0x1.23456p-2));
  test(STR("1#234560p-1"), loc, STR("{:.6La}"), F(0x1.23456p-1));
  test(STR("1#234560p+0"), loc, STR("{:.6La}"), F(0x1.23456p0));
  test(STR("1#234560p+1"), loc, STR("{:.6La}"), F(0x1.23456p+1));
  test(STR("1#234560p+2"), loc, STR("{:.6La}"), F(0x1.23456p+2));
  test(STR("1#234560p+3"), loc, STR("{:.6La}"), F(0x1.23456p+3));
  test(STR("1#234560p+20"), loc, STR("{:.6La}"), F(0x1.23456p+20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(STR("1.234560p+3$$$"), STR("{:$<14.6La}"), F(0x1.23456p3));
  test(STR("$$$1.234560p+3"), STR("{:$>14.6La}"), F(0x1.23456p3));
  test(STR("$1.234560p+3$$"), STR("{:$^14.6La}"), F(0x1.23456p3));
  test(STR("0001.234560p+3"), STR("{:014.6La}"), F(0x1.23456p3));
  test(STR("-1.234560p+3$$$"), STR("{:$<15.6La}"), F(-0x1.23456p3));
  test(STR("$$$-1.234560p+3"), STR("{:$>15.6La}"), F(-0x1.23456p3));
  test(STR("$-1.234560p+3$$"), STR("{:$^15.6La}"), F(-0x1.23456p3));
  test(STR("-0001.234560p+3"), STR("{:015.6La}"), F(-0x1.23456p3));

  std::locale::global(loc);
  test(STR("1#234560p+3$$$"), STR("{:$<14.6La}"), F(0x1.23456p3));
  test(STR("$$$1#234560p+3"), STR("{:$>14.6La}"), F(0x1.23456p3));
  test(STR("$1#234560p+3$$"), STR("{:$^14.6La}"), F(0x1.23456p3));
  test(STR("0001#234560p+3"), STR("{:014.6La}"), F(0x1.23456p3));
  test(STR("-1#234560p+3$$$"), STR("{:$<15.6La}"), F(-0x1.23456p3));
  test(STR("$$$-1#234560p+3"), STR("{:$>15.6La}"), F(-0x1.23456p3));
  test(STR("$-1#234560p+3$$"), STR("{:$^15.6La}"), F(-0x1.23456p3));
  test(STR("-0001#234560p+3"), STR("{:015.6La}"), F(-0x1.23456p3));

  test(STR("1.234560p+3$$$"), en_US, STR("{:$<14.6La}"), F(0x1.23456p3));
  test(STR("$$$1.234560p+3"), en_US, STR("{:$>14.6La}"), F(0x1.23456p3));
  test(STR("$1.234560p+3$$"), en_US, STR("{:$^14.6La}"), F(0x1.23456p3));
  test(STR("0001.234560p+3"), en_US, STR("{:014.6La}"), F(0x1.23456p3));
  test(STR("-1.234560p+3$$$"), en_US, STR("{:$<15.6La}"), F(-0x1.23456p3));
  test(STR("$$$-1.234560p+3"), en_US, STR("{:$>15.6La}"), F(-0x1.23456p3));
  test(STR("$-1.234560p+3$$"), en_US, STR("{:$^15.6La}"), F(-0x1.23456p3));
  test(STR("-0001.234560p+3"), en_US, STR("{:015.6La}"), F(-0x1.23456p3));

  std::locale::global(en_US);
  test(STR("1#234560p+3$$$"), loc, STR("{:$<14.6La}"), F(0x1.23456p3));
  test(STR("$$$1#234560p+3"), loc, STR("{:$>14.6La}"), F(0x1.23456p3));
  test(STR("$1#234560p+3$$"), loc, STR("{:$^14.6La}"), F(0x1.23456p3));
  test(STR("0001#234560p+3"), loc, STR("{:014.6La}"), F(0x1.23456p3));
  test(STR("-1#234560p+3$$$"), loc, STR("{:$<15.6La}"), F(-0x1.23456p3));
  test(STR("$$$-1#234560p+3"), loc, STR("{:$>15.6La}"), F(-0x1.23456p3));
  test(STR("$-1#234560p+3$$"), loc, STR("{:$^15.6La}"), F(-0x1.23456p3));
  test(STR("-0001#234560p+3"), loc, STR("{:015.6La}"), F(-0x1.23456p3));
}

template <class F, class CharT>
void test_floating_point_hex_upper_case_precision() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(STR("1.234560P-3"), STR("{:.6LA}"), F(0x1.23456p-3));
  test(STR("1.234560P-2"), STR("{:.6LA}"), F(0x1.23456p-2));
  test(STR("1.234560P-1"), STR("{:.6LA}"), F(0x1.23456p-1));
  test(STR("1.234560P+0"), STR("{:.6LA}"), F(0x1.23456p0));
  test(STR("1.234560P+1"), STR("{:.6LA}"), F(0x1.23456p+1));
  test(STR("1.234560P+2"), STR("{:.6LA}"), F(0x1.23456p+2));
  test(STR("1.234560P+3"), STR("{:.6LA}"), F(0x1.23456p+3));
  test(STR("1.234560P+20"), STR("{:.6LA}"), F(0x1.23456p+20));

  std::locale::global(loc);
  test(STR("1#234560P-3"), STR("{:.6LA}"), F(0x1.23456p-3));
  test(STR("1#234560P-2"), STR("{:.6LA}"), F(0x1.23456p-2));
  test(STR("1#234560P-1"), STR("{:.6LA}"), F(0x1.23456p-1));
  test(STR("1#234560P+0"), STR("{:.6LA}"), F(0x1.23456p0));
  test(STR("1#234560P+1"), STR("{:.6LA}"), F(0x1.23456p+1));
  test(STR("1#234560P+2"), STR("{:.6LA}"), F(0x1.23456p+2));
  test(STR("1#234560P+3"), STR("{:.6LA}"), F(0x1.23456p+3));
  test(STR("1#234560P+20"), STR("{:.6LA}"), F(0x1.23456p+20));

  test(STR("1.234560P-3"), en_US, STR("{:.6LA}"), F(0x1.23456p-3));
  test(STR("1.234560P-2"), en_US, STR("{:.6LA}"), F(0x1.23456p-2));
  test(STR("1.234560P-1"), en_US, STR("{:.6LA}"), F(0x1.23456p-1));
  test(STR("1.234560P+0"), en_US, STR("{:.6LA}"), F(0x1.23456p0));
  test(STR("1.234560P+1"), en_US, STR("{:.6LA}"), F(0x1.23456p+1));
  test(STR("1.234560P+2"), en_US, STR("{:.6LA}"), F(0x1.23456p+2));
  test(STR("1.234560P+3"), en_US, STR("{:.6LA}"), F(0x1.23456p+3));
  test(STR("1.234560P+20"), en_US, STR("{:.6LA}"), F(0x1.23456p+20));

  std::locale::global(en_US);
  test(STR("1#234560P-3"), loc, STR("{:.6LA}"), F(0x1.23456p-3));
  test(STR("1#234560P-2"), loc, STR("{:.6LA}"), F(0x1.23456p-2));
  test(STR("1#234560P-1"), loc, STR("{:.6LA}"), F(0x1.23456p-1));
  test(STR("1#234560P+0"), loc, STR("{:.6LA}"), F(0x1.23456p0));
  test(STR("1#234560P+1"), loc, STR("{:.6LA}"), F(0x1.23456p+1));
  test(STR("1#234560P+2"), loc, STR("{:.6LA}"), F(0x1.23456p+2));
  test(STR("1#234560P+3"), loc, STR("{:.6LA}"), F(0x1.23456p+3));
  test(STR("1#234560P+20"), loc, STR("{:.6LA}"), F(0x1.23456p+20));

  // *** Fill, align, zero Padding ***
  std::locale::global(en_US);
  test(STR("1.234560P+3$$$"), STR("{:$<14.6LA}"), F(0x1.23456p3));
  test(STR("$$$1.234560P+3"), STR("{:$>14.6LA}"), F(0x1.23456p3));
  test(STR("$1.234560P+3$$"), STR("{:$^14.6LA}"), F(0x1.23456p3));
  test(STR("0001.234560P+3"), STR("{:014.6LA}"), F(0x1.23456p3));
  test(STR("-1.234560P+3$$$"), STR("{:$<15.6LA}"), F(-0x1.23456p3));
  test(STR("$$$-1.234560P+3"), STR("{:$>15.6LA}"), F(-0x1.23456p3));
  test(STR("$-1.234560P+3$$"), STR("{:$^15.6LA}"), F(-0x1.23456p3));
  test(STR("-0001.234560P+3"), STR("{:015.6LA}"), F(-0x1.23456p3));

  std::locale::global(loc);
  test(STR("1#234560P+3$$$"), STR("{:$<14.6LA}"), F(0x1.23456p3));
  test(STR("$$$1#234560P+3"), STR("{:$>14.6LA}"), F(0x1.23456p3));
  test(STR("$1#234560P+3$$"), STR("{:$^14.6LA}"), F(0x1.23456p3));
  test(STR("0001#234560P+3"), STR("{:014.6LA}"), F(0x1.23456p3));
  test(STR("-1#234560P+3$$$"), STR("{:$<15.6LA}"), F(-0x1.23456p3));
  test(STR("$$$-1#234560P+3"), STR("{:$>15.6LA}"), F(-0x1.23456p3));
  test(STR("$-1#234560P+3$$"), STR("{:$^15.6LA}"), F(-0x1.23456p3));
  test(STR("-0001#234560P+3"), STR("{:015.6LA}"), F(-0x1.23456p3));

  test(STR("1.234560P+3$$$"), en_US, STR("{:$<14.6LA}"), F(0x1.23456p3));
  test(STR("$$$1.234560P+3"), en_US, STR("{:$>14.6LA}"), F(0x1.23456p3));
  test(STR("$1.234560P+3$$"), en_US, STR("{:$^14.6LA}"), F(0x1.23456p3));
  test(STR("0001.234560P+3"), en_US, STR("{:014.6LA}"), F(0x1.23456p3));
  test(STR("-1.234560P+3$$$"), en_US, STR("{:$<15.6LA}"), F(-0x1.23456p3));
  test(STR("$$$-1.234560P+3"), en_US, STR("{:$>15.6LA}"), F(-0x1.23456p3));
  test(STR("$-1.234560P+3$$"), en_US, STR("{:$^15.6LA}"), F(-0x1.23456p3));
  test(STR("-0001.234560P+3"), en_US, STR("{:015.6LA}"), F(-0x1.23456p3));

  std::locale::global(en_US);
  test(STR("1#234560P+3$$$"), loc, STR("{:$<14.6LA}"), F(0x1.23456p3));
  test(STR("$$$1#234560P+3"), loc, STR("{:$>14.6LA}"), F(0x1.23456p3));
  test(STR("$1#234560P+3$$"), loc, STR("{:$^14.6LA}"), F(0x1.23456p3));
  test(STR("0001#234560P+3"), loc, STR("{:014.6LA}"), F(0x1.23456p3));
  test(STR("-1#234560P+3$$$"), loc, STR("{:$<15.6LA}"), F(-0x1.23456p3));
  test(STR("$$$-1#234560P+3"), loc, STR("{:$>15.6LA}"), F(-0x1.23456p3));
  test(STR("$-1#234560P+3$$"), loc, STR("{:$^15.6LA}"), F(-0x1.23456p3));
  test(STR("-0001#234560P+3"), loc, STR("{:015.6LA}"), F(-0x1.23456p3));
}

template <class F, class CharT>
void test_floating_point_scientific_lower_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(STR("1.234567e-03"), STR("{:.6Le}"), F(1.234567e-3));
  test(STR("1.234567e-02"), STR("{:.6Le}"), F(1.234567e-2));
  test(STR("1.234567e-01"), STR("{:.6Le}"), F(1.234567e-1));
  test(STR("1.234567e+00"), STR("{:.6Le}"), F(1.234567e0));
  test(STR("1.234567e+01"), STR("{:.6Le}"), F(1.234567e1));
  test(STR("1.234567e+02"), STR("{:.6Le}"), F(1.234567e2));
  test(STR("1.234567e+03"), STR("{:.6Le}"), F(1.234567e3));
  test(STR("1.234567e+20"), STR("{:.6Le}"), F(1.234567e20));
  test(STR("-1.234567e-03"), STR("{:.6Le}"), F(-1.234567e-3));
  test(STR("-1.234567e-02"), STR("{:.6Le}"), F(-1.234567e-2));
  test(STR("-1.234567e-01"), STR("{:.6Le}"), F(-1.234567e-1));
  test(STR("-1.234567e+00"), STR("{:.6Le}"), F(-1.234567e0));
  test(STR("-1.234567e+01"), STR("{:.6Le}"), F(-1.234567e1));
  test(STR("-1.234567e+02"), STR("{:.6Le}"), F(-1.234567e2));
  test(STR("-1.234567e+03"), STR("{:.6Le}"), F(-1.234567e3));
  test(STR("-1.234567e+20"), STR("{:.6Le}"), F(-1.234567e20));

  std::locale::global(loc);
  test(STR("1#234567e-03"), STR("{:.6Le}"), F(1.234567e-3));
  test(STR("1#234567e-02"), STR("{:.6Le}"), F(1.234567e-2));
  test(STR("1#234567e-01"), STR("{:.6Le}"), F(1.234567e-1));
  test(STR("1#234567e+00"), STR("{:.6Le}"), F(1.234567e0));
  test(STR("1#234567e+01"), STR("{:.6Le}"), F(1.234567e1));
  test(STR("1#234567e+02"), STR("{:.6Le}"), F(1.234567e2));
  test(STR("1#234567e+03"), STR("{:.6Le}"), F(1.234567e3));
  test(STR("1#234567e+20"), STR("{:.6Le}"), F(1.234567e20));
  test(STR("-1#234567e-03"), STR("{:.6Le}"), F(-1.234567e-3));
  test(STR("-1#234567e-02"), STR("{:.6Le}"), F(-1.234567e-2));
  test(STR("-1#234567e-01"), STR("{:.6Le}"), F(-1.234567e-1));
  test(STR("-1#234567e+00"), STR("{:.6Le}"), F(-1.234567e0));
  test(STR("-1#234567e+01"), STR("{:.6Le}"), F(-1.234567e1));
  test(STR("-1#234567e+02"), STR("{:.6Le}"), F(-1.234567e2));
  test(STR("-1#234567e+03"), STR("{:.6Le}"), F(-1.234567e3));
  test(STR("-1#234567e+20"), STR("{:.6Le}"), F(-1.234567e20));

  test(STR("1.234567e-03"), en_US, STR("{:.6Le}"), F(1.234567e-3));
  test(STR("1.234567e-02"), en_US, STR("{:.6Le}"), F(1.234567e-2));
  test(STR("1.234567e-01"), en_US, STR("{:.6Le}"), F(1.234567e-1));
  test(STR("1.234567e+00"), en_US, STR("{:.6Le}"), F(1.234567e0));
  test(STR("1.234567e+01"), en_US, STR("{:.6Le}"), F(1.234567e1));
  test(STR("1.234567e+02"), en_US, STR("{:.6Le}"), F(1.234567e2));
  test(STR("1.234567e+03"), en_US, STR("{:.6Le}"), F(1.234567e3));
  test(STR("1.234567e+20"), en_US, STR("{:.6Le}"), F(1.234567e20));
  test(STR("-1.234567e-03"), en_US, STR("{:.6Le}"), F(-1.234567e-3));
  test(STR("-1.234567e-02"), en_US, STR("{:.6Le}"), F(-1.234567e-2));
  test(STR("-1.234567e-01"), en_US, STR("{:.6Le}"), F(-1.234567e-1));
  test(STR("-1.234567e+00"), en_US, STR("{:.6Le}"), F(-1.234567e0));
  test(STR("-1.234567e+01"), en_US, STR("{:.6Le}"), F(-1.234567e1));
  test(STR("-1.234567e+02"), en_US, STR("{:.6Le}"), F(-1.234567e2));
  test(STR("-1.234567e+03"), en_US, STR("{:.6Le}"), F(-1.234567e3));
  test(STR("-1.234567e+20"), en_US, STR("{:.6Le}"), F(-1.234567e20));

  std::locale::global(en_US);
  test(STR("1#234567e-03"), loc, STR("{:.6Le}"), F(1.234567e-3));
  test(STR("1#234567e-02"), loc, STR("{:.6Le}"), F(1.234567e-2));
  test(STR("1#234567e-01"), loc, STR("{:.6Le}"), F(1.234567e-1));
  test(STR("1#234567e+00"), loc, STR("{:.6Le}"), F(1.234567e0));
  test(STR("1#234567e+01"), loc, STR("{:.6Le}"), F(1.234567e1));
  test(STR("1#234567e+02"), loc, STR("{:.6Le}"), F(1.234567e2));
  test(STR("1#234567e+03"), loc, STR("{:.6Le}"), F(1.234567e3));
  test(STR("1#234567e+20"), loc, STR("{:.6Le}"), F(1.234567e20));
  test(STR("-1#234567e-03"), loc, STR("{:.6Le}"), F(-1.234567e-3));
  test(STR("-1#234567e-02"), loc, STR("{:.6Le}"), F(-1.234567e-2));
  test(STR("-1#234567e-01"), loc, STR("{:.6Le}"), F(-1.234567e-1));
  test(STR("-1#234567e+00"), loc, STR("{:.6Le}"), F(-1.234567e0));
  test(STR("-1#234567e+01"), loc, STR("{:.6Le}"), F(-1.234567e1));
  test(STR("-1#234567e+02"), loc, STR("{:.6Le}"), F(-1.234567e2));
  test(STR("-1#234567e+03"), loc, STR("{:.6Le}"), F(-1.234567e3));
  test(STR("-1#234567e+20"), loc, STR("{:.6Le}"), F(-1.234567e20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(STR("1.234567e+03$$$"), STR("{:$<15.6Le}"), F(1.234567e3));
  test(STR("$$$1.234567e+03"), STR("{:$>15.6Le}"), F(1.234567e3));
  test(STR("$1.234567e+03$$"), STR("{:$^15.6Le}"), F(1.234567e3));
  test(STR("0001.234567e+03"), STR("{:015.6Le}"), F(1.234567e3));
  test(STR("-1.234567e+03$$$"), STR("{:$<16.6Le}"), F(-1.234567e3));
  test(STR("$$$-1.234567e+03"), STR("{:$>16.6Le}"), F(-1.234567e3));
  test(STR("$-1.234567e+03$$"), STR("{:$^16.6Le}"), F(-1.234567e3));
  test(STR("-0001.234567e+03"), STR("{:016.6Le}"), F(-1.234567e3));

  std::locale::global(loc);
  test(STR("1#234567e+03$$$"), STR("{:$<15.6Le}"), F(1.234567e3));
  test(STR("$$$1#234567e+03"), STR("{:$>15.6Le}"), F(1.234567e3));
  test(STR("$1#234567e+03$$"), STR("{:$^15.6Le}"), F(1.234567e3));
  test(STR("0001#234567e+03"), STR("{:015.6Le}"), F(1.234567e3));
  test(STR("-1#234567e+03$$$"), STR("{:$<16.6Le}"), F(-1.234567e3));
  test(STR("$$$-1#234567e+03"), STR("{:$>16.6Le}"), F(-1.234567e3));
  test(STR("$-1#234567e+03$$"), STR("{:$^16.6Le}"), F(-1.234567e3));
  test(STR("-0001#234567e+03"), STR("{:016.6Le}"), F(-1.234567e3));

  test(STR("1.234567e+03$$$"), en_US, STR("{:$<15.6Le}"), F(1.234567e3));
  test(STR("$$$1.234567e+03"), en_US, STR("{:$>15.6Le}"), F(1.234567e3));
  test(STR("$1.234567e+03$$"), en_US, STR("{:$^15.6Le}"), F(1.234567e3));
  test(STR("0001.234567e+03"), en_US, STR("{:015.6Le}"), F(1.234567e3));
  test(STR("-1.234567e+03$$$"), en_US, STR("{:$<16.6Le}"), F(-1.234567e3));
  test(STR("$$$-1.234567e+03"), en_US, STR("{:$>16.6Le}"), F(-1.234567e3));
  test(STR("$-1.234567e+03$$"), en_US, STR("{:$^16.6Le}"), F(-1.234567e3));
  test(STR("-0001.234567e+03"), en_US, STR("{:016.6Le}"), F(-1.234567e3));

  std::locale::global(en_US);
  test(STR("1#234567e+03$$$"), loc, STR("{:$<15.6Le}"), F(1.234567e3));
  test(STR("$$$1#234567e+03"), loc, STR("{:$>15.6Le}"), F(1.234567e3));
  test(STR("$1#234567e+03$$"), loc, STR("{:$^15.6Le}"), F(1.234567e3));
  test(STR("0001#234567e+03"), loc, STR("{:015.6Le}"), F(1.234567e3));
  test(STR("-1#234567e+03$$$"), loc, STR("{:$<16.6Le}"), F(-1.234567e3));
  test(STR("$$$-1#234567e+03"), loc, STR("{:$>16.6Le}"), F(-1.234567e3));
  test(STR("$-1#234567e+03$$"), loc, STR("{:$^16.6Le}"), F(-1.234567e3));
  test(STR("-0001#234567e+03"), loc, STR("{:016.6Le}"), F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_scientific_upper_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(STR("1.234567E-03"), STR("{:.6LE}"), F(1.234567e-3));
  test(STR("1.234567E-02"), STR("{:.6LE}"), F(1.234567e-2));
  test(STR("1.234567E-01"), STR("{:.6LE}"), F(1.234567e-1));
  test(STR("1.234567E+00"), STR("{:.6LE}"), F(1.234567e0));
  test(STR("1.234567E+01"), STR("{:.6LE}"), F(1.234567e1));
  test(STR("1.234567E+02"), STR("{:.6LE}"), F(1.234567e2));
  test(STR("1.234567E+03"), STR("{:.6LE}"), F(1.234567e3));
  test(STR("1.234567E+20"), STR("{:.6LE}"), F(1.234567e20));
  test(STR("-1.234567E-03"), STR("{:.6LE}"), F(-1.234567e-3));
  test(STR("-1.234567E-02"), STR("{:.6LE}"), F(-1.234567e-2));
  test(STR("-1.234567E-01"), STR("{:.6LE}"), F(-1.234567e-1));
  test(STR("-1.234567E+00"), STR("{:.6LE}"), F(-1.234567e0));
  test(STR("-1.234567E+01"), STR("{:.6LE}"), F(-1.234567e1));
  test(STR("-1.234567E+02"), STR("{:.6LE}"), F(-1.234567e2));
  test(STR("-1.234567E+03"), STR("{:.6LE}"), F(-1.234567e3));
  test(STR("-1.234567E+20"), STR("{:.6LE}"), F(-1.234567e20));

  std::locale::global(loc);
  test(STR("1#234567E-03"), STR("{:.6LE}"), F(1.234567e-3));
  test(STR("1#234567E-02"), STR("{:.6LE}"), F(1.234567e-2));
  test(STR("1#234567E-01"), STR("{:.6LE}"), F(1.234567e-1));
  test(STR("1#234567E+00"), STR("{:.6LE}"), F(1.234567e0));
  test(STR("1#234567E+01"), STR("{:.6LE}"), F(1.234567e1));
  test(STR("1#234567E+02"), STR("{:.6LE}"), F(1.234567e2));
  test(STR("1#234567E+03"), STR("{:.6LE}"), F(1.234567e3));
  test(STR("1#234567E+20"), STR("{:.6LE}"), F(1.234567e20));
  test(STR("-1#234567E-03"), STR("{:.6LE}"), F(-1.234567e-3));
  test(STR("-1#234567E-02"), STR("{:.6LE}"), F(-1.234567e-2));
  test(STR("-1#234567E-01"), STR("{:.6LE}"), F(-1.234567e-1));
  test(STR("-1#234567E+00"), STR("{:.6LE}"), F(-1.234567e0));
  test(STR("-1#234567E+01"), STR("{:.6LE}"), F(-1.234567e1));
  test(STR("-1#234567E+02"), STR("{:.6LE}"), F(-1.234567e2));
  test(STR("-1#234567E+03"), STR("{:.6LE}"), F(-1.234567e3));
  test(STR("-1#234567E+20"), STR("{:.6LE}"), F(-1.234567e20));

  test(STR("1.234567E-03"), en_US, STR("{:.6LE}"), F(1.234567e-3));
  test(STR("1.234567E-02"), en_US, STR("{:.6LE}"), F(1.234567e-2));
  test(STR("1.234567E-01"), en_US, STR("{:.6LE}"), F(1.234567e-1));
  test(STR("1.234567E+00"), en_US, STR("{:.6LE}"), F(1.234567e0));
  test(STR("1.234567E+01"), en_US, STR("{:.6LE}"), F(1.234567e1));
  test(STR("1.234567E+02"), en_US, STR("{:.6LE}"), F(1.234567e2));
  test(STR("1.234567E+03"), en_US, STR("{:.6LE}"), F(1.234567e3));
  test(STR("1.234567E+20"), en_US, STR("{:.6LE}"), F(1.234567e20));
  test(STR("-1.234567E-03"), en_US, STR("{:.6LE}"), F(-1.234567e-3));
  test(STR("-1.234567E-02"), en_US, STR("{:.6LE}"), F(-1.234567e-2));
  test(STR("-1.234567E-01"), en_US, STR("{:.6LE}"), F(-1.234567e-1));
  test(STR("-1.234567E+00"), en_US, STR("{:.6LE}"), F(-1.234567e0));
  test(STR("-1.234567E+01"), en_US, STR("{:.6LE}"), F(-1.234567e1));
  test(STR("-1.234567E+02"), en_US, STR("{:.6LE}"), F(-1.234567e2));
  test(STR("-1.234567E+03"), en_US, STR("{:.6LE}"), F(-1.234567e3));
  test(STR("-1.234567E+20"), en_US, STR("{:.6LE}"), F(-1.234567e20));

  std::locale::global(en_US);
  test(STR("1#234567E-03"), loc, STR("{:.6LE}"), F(1.234567e-3));
  test(STR("1#234567E-02"), loc, STR("{:.6LE}"), F(1.234567e-2));
  test(STR("1#234567E-01"), loc, STR("{:.6LE}"), F(1.234567e-1));
  test(STR("1#234567E+00"), loc, STR("{:.6LE}"), F(1.234567e0));
  test(STR("1#234567E+01"), loc, STR("{:.6LE}"), F(1.234567e1));
  test(STR("1#234567E+02"), loc, STR("{:.6LE}"), F(1.234567e2));
  test(STR("1#234567E+03"), loc, STR("{:.6LE}"), F(1.234567e3));
  test(STR("1#234567E+20"), loc, STR("{:.6LE}"), F(1.234567e20));
  test(STR("-1#234567E-03"), loc, STR("{:.6LE}"), F(-1.234567e-3));
  test(STR("-1#234567E-02"), loc, STR("{:.6LE}"), F(-1.234567e-2));
  test(STR("-1#234567E-01"), loc, STR("{:.6LE}"), F(-1.234567e-1));
  test(STR("-1#234567E+00"), loc, STR("{:.6LE}"), F(-1.234567e0));
  test(STR("-1#234567E+01"), loc, STR("{:.6LE}"), F(-1.234567e1));
  test(STR("-1#234567E+02"), loc, STR("{:.6LE}"), F(-1.234567e2));
  test(STR("-1#234567E+03"), loc, STR("{:.6LE}"), F(-1.234567e3));
  test(STR("-1#234567E+20"), loc, STR("{:.6LE}"), F(-1.234567e20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(STR("1.234567E+03$$$"), STR("{:$<15.6LE}"), F(1.234567e3));
  test(STR("$$$1.234567E+03"), STR("{:$>15.6LE}"), F(1.234567e3));
  test(STR("$1.234567E+03$$"), STR("{:$^15.6LE}"), F(1.234567e3));
  test(STR("0001.234567E+03"), STR("{:015.6LE}"), F(1.234567e3));
  test(STR("-1.234567E+03$$$"), STR("{:$<16.6LE}"), F(-1.234567e3));
  test(STR("$$$-1.234567E+03"), STR("{:$>16.6LE}"), F(-1.234567e3));
  test(STR("$-1.234567E+03$$"), STR("{:$^16.6LE}"), F(-1.234567e3));
  test(STR("-0001.234567E+03"), STR("{:016.6LE}"), F(-1.234567e3));

  std::locale::global(loc);
  test(STR("1#234567E+03$$$"), STR("{:$<15.6LE}"), F(1.234567e3));
  test(STR("$$$1#234567E+03"), STR("{:$>15.6LE}"), F(1.234567e3));
  test(STR("$1#234567E+03$$"), STR("{:$^15.6LE}"), F(1.234567e3));
  test(STR("0001#234567E+03"), STR("{:015.6LE}"), F(1.234567e3));
  test(STR("-1#234567E+03$$$"), STR("{:$<16.6LE}"), F(-1.234567e3));
  test(STR("$$$-1#234567E+03"), STR("{:$>16.6LE}"), F(-1.234567e3));
  test(STR("$-1#234567E+03$$"), STR("{:$^16.6LE}"), F(-1.234567e3));
  test(STR("-0001#234567E+03"), STR("{:016.6LE}"), F(-1.234567e3));

  test(STR("1.234567E+03$$$"), en_US, STR("{:$<15.6LE}"), F(1.234567e3));
  test(STR("$$$1.234567E+03"), en_US, STR("{:$>15.6LE}"), F(1.234567e3));
  test(STR("$1.234567E+03$$"), en_US, STR("{:$^15.6LE}"), F(1.234567e3));
  test(STR("0001.234567E+03"), en_US, STR("{:015.6LE}"), F(1.234567e3));
  test(STR("-1.234567E+03$$$"), en_US, STR("{:$<16.6LE}"), F(-1.234567e3));
  test(STR("$$$-1.234567E+03"), en_US, STR("{:$>16.6LE}"), F(-1.234567e3));
  test(STR("$-1.234567E+03$$"), en_US, STR("{:$^16.6LE}"), F(-1.234567e3));
  test(STR("-0001.234567E+03"), en_US, STR("{:016.6LE}"), F(-1.234567e3));

  std::locale::global(en_US);
  test(STR("1#234567E+03$$$"), loc, STR("{:$<15.6LE}"), F(1.234567e3));
  test(STR("$$$1#234567E+03"), loc, STR("{:$>15.6LE}"), F(1.234567e3));
  test(STR("$1#234567E+03$$"), loc, STR("{:$^15.6LE}"), F(1.234567e3));
  test(STR("0001#234567E+03"), loc, STR("{:015.6LE}"), F(1.234567e3));
  test(STR("-1#234567E+03$$$"), loc, STR("{:$<16.6LE}"), F(-1.234567e3));
  test(STR("$$$-1#234567E+03"), loc, STR("{:$>16.6LE}"), F(-1.234567e3));
  test(STR("$-1#234567E+03$$"), loc, STR("{:$^16.6LE}"), F(-1.234567e3));
  test(STR("-0001#234567E+03"), loc, STR("{:016.6LE}"), F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_fixed_lower_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(STR("0.000001"), STR("{:.6Lf}"), F(1.234567e-6));
  test(STR("0.000012"), STR("{:.6Lf}"), F(1.234567e-5));
  test(STR("0.000123"), STR("{:.6Lf}"), F(1.234567e-4));
  test(STR("0.001235"), STR("{:.6Lf}"), F(1.234567e-3));
  test(STR("0.012346"), STR("{:.6Lf}"), F(1.234567e-2));
  test(STR("0.123457"), STR("{:.6Lf}"), F(1.234567e-1));
  test(STR("1.234567"), STR("{:.6Lf}"), F(1.234567e0));
  test(STR("12.345670"), STR("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("123.456700"), STR("{:.6Lf}"), F(1.234567e2));
    test(STR("1,234.567000"), STR("{:.6Lf}"), F(1.234567e3));
    test(STR("12,345.670000"), STR("{:.6Lf}"), F(1.234567e4));
    test(STR("123,456.700000"), STR("{:.6Lf}"), F(1.234567e5));
    test(STR("1,234,567.000000"), STR("{:.6Lf}"), F(1.234567e6));
    test(STR("12,345,670.000000"), STR("{:.6Lf}"), F(1.234567e7));
    test(STR("123,456,700,000,000,000,000.000000"), STR("{:.6Lf}"), F(1.234567e20));
  }
  test(STR("-0.000001"), STR("{:.6Lf}"), F(-1.234567e-6));
  test(STR("-0.000012"), STR("{:.6Lf}"), F(-1.234567e-5));
  test(STR("-0.000123"), STR("{:.6Lf}"), F(-1.234567e-4));
  test(STR("-0.001235"), STR("{:.6Lf}"), F(-1.234567e-3));
  test(STR("-0.012346"), STR("{:.6Lf}"), F(-1.234567e-2));
  test(STR("-0.123457"), STR("{:.6Lf}"), F(-1.234567e-1));
  test(STR("-1.234567"), STR("{:.6Lf}"), F(-1.234567e0));
  test(STR("-12.345670"), STR("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("-123.456700"), STR("{:.6Lf}"), F(-1.234567e2));
    test(STR("-1,234.567000"), STR("{:.6Lf}"), F(-1.234567e3));
    test(STR("-12,345.670000"), STR("{:.6Lf}"), F(-1.234567e4));
    test(STR("-123,456.700000"), STR("{:.6Lf}"), F(-1.234567e5));
    test(STR("-1,234,567.000000"), STR("{:.6Lf}"), F(-1.234567e6));
    test(STR("-12,345,670.000000"), STR("{:.6Lf}"), F(-1.234567e7));
    test(STR("-123,456,700,000,000,000,000.000000"), STR("{:.6Lf}"), F(-1.234567e20));
  }

  std::locale::global(loc);
  test(STR("0#000001"), STR("{:.6Lf}"), F(1.234567e-6));
  test(STR("0#000012"), STR("{:.6Lf}"), F(1.234567e-5));
  test(STR("0#000123"), STR("{:.6Lf}"), F(1.234567e-4));
  test(STR("0#001235"), STR("{:.6Lf}"), F(1.234567e-3));
  test(STR("0#012346"), STR("{:.6Lf}"), F(1.234567e-2));
  test(STR("0#123457"), STR("{:.6Lf}"), F(1.234567e-1));
  test(STR("1#234567"), STR("{:.6Lf}"), F(1.234567e0));
  test(STR("1_2#345670"), STR("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("12_3#456700"), STR("{:.6Lf}"), F(1.234567e2));
    test(STR("1_23_4#567000"), STR("{:.6Lf}"), F(1.234567e3));
    test(STR("12_34_5#670000"), STR("{:.6Lf}"), F(1.234567e4));
    test(STR("123_45_6#700000"), STR("{:.6Lf}"), F(1.234567e5));
    test(STR("1_234_56_7#000000"), STR("{:.6Lf}"), F(1.234567e6));
    test(STR("12_345_67_0#000000"), STR("{:.6Lf}"), F(1.234567e7));
    test(STR("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), STR("{:.6Lf}"), F(1.234567e20));
  }
  test(STR("-0#000001"), STR("{:.6Lf}"), F(-1.234567e-6));
  test(STR("-0#000012"), STR("{:.6Lf}"), F(-1.234567e-5));
  test(STR("-0#000123"), STR("{:.6Lf}"), F(-1.234567e-4));
  test(STR("-0#001235"), STR("{:.6Lf}"), F(-1.234567e-3));
  test(STR("-0#012346"), STR("{:.6Lf}"), F(-1.234567e-2));
  test(STR("-0#123457"), STR("{:.6Lf}"), F(-1.234567e-1));
  test(STR("-1#234567"), STR("{:.6Lf}"), F(-1.234567e0));
  test(STR("-1_2#345670"), STR("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("-12_3#456700"), STR("{:.6Lf}"), F(-1.234567e2));
    test(STR("-1_23_4#567000"), STR("{:.6Lf}"), F(-1.234567e3));
    test(STR("-12_34_5#670000"), STR("{:.6Lf}"), F(-1.234567e4));
    test(STR("-123_45_6#700000"), STR("{:.6Lf}"), F(-1.234567e5));
    test(STR("-1_234_56_7#000000"), STR("{:.6Lf}"), F(-1.234567e6));
    test(STR("-12_345_67_0#000000"), STR("{:.6Lf}"), F(-1.234567e7));
    test(STR("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), STR("{:.6Lf}"), F(-1.234567e20));
  }

  test(STR("0.000001"), en_US, STR("{:.6Lf}"), F(1.234567e-6));
  test(STR("0.000012"), en_US, STR("{:.6Lf}"), F(1.234567e-5));
  test(STR("0.000123"), en_US, STR("{:.6Lf}"), F(1.234567e-4));
  test(STR("0.001235"), en_US, STR("{:.6Lf}"), F(1.234567e-3));
  test(STR("0.012346"), en_US, STR("{:.6Lf}"), F(1.234567e-2));
  test(STR("0.123457"), en_US, STR("{:.6Lf}"), F(1.234567e-1));
  test(STR("1.234567"), en_US, STR("{:.6Lf}"), F(1.234567e0));
  test(STR("12.345670"), en_US, STR("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("123.456700"), en_US, STR("{:.6Lf}"), F(1.234567e2));
    test(STR("1,234.567000"), en_US, STR("{:.6Lf}"), F(1.234567e3));
    test(STR("12,345.670000"), en_US, STR("{:.6Lf}"), F(1.234567e4));
    test(STR("123,456.700000"), en_US, STR("{:.6Lf}"), F(1.234567e5));
    test(STR("1,234,567.000000"), en_US, STR("{:.6Lf}"), F(1.234567e6));
    test(STR("12,345,670.000000"), en_US, STR("{:.6Lf}"), F(1.234567e7));
    test(STR("123,456,700,000,000,000,000.000000"), en_US, STR("{:.6Lf}"), F(1.234567e20));
  }
  test(STR("-0.000001"), en_US, STR("{:.6Lf}"), F(-1.234567e-6));
  test(STR("-0.000012"), en_US, STR("{:.6Lf}"), F(-1.234567e-5));
  test(STR("-0.000123"), en_US, STR("{:.6Lf}"), F(-1.234567e-4));
  test(STR("-0.001235"), en_US, STR("{:.6Lf}"), F(-1.234567e-3));
  test(STR("-0.012346"), en_US, STR("{:.6Lf}"), F(-1.234567e-2));
  test(STR("-0.123457"), en_US, STR("{:.6Lf}"), F(-1.234567e-1));
  test(STR("-1.234567"), en_US, STR("{:.6Lf}"), F(-1.234567e0));
  test(STR("-12.345670"), en_US, STR("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("-123.456700"), en_US, STR("{:.6Lf}"), F(-1.234567e2));
    test(STR("-1,234.567000"), en_US, STR("{:.6Lf}"), F(-1.234567e3));
    test(STR("-12,345.670000"), en_US, STR("{:.6Lf}"), F(-1.234567e4));
    test(STR("-123,456.700000"), en_US, STR("{:.6Lf}"), F(-1.234567e5));
    test(STR("-1,234,567.000000"), en_US, STR("{:.6Lf}"), F(-1.234567e6));
    test(STR("-12,345,670.000000"), en_US, STR("{:.6Lf}"), F(-1.234567e7));
    test(STR("-123,456,700,000,000,000,000.000000"), en_US, STR("{:.6Lf}"), F(-1.234567e20));
  }

  std::locale::global(en_US);
  test(STR("0#000001"), loc, STR("{:.6Lf}"), F(1.234567e-6));
  test(STR("0#000012"), loc, STR("{:.6Lf}"), F(1.234567e-5));
  test(STR("0#000123"), loc, STR("{:.6Lf}"), F(1.234567e-4));
  test(STR("0#001235"), loc, STR("{:.6Lf}"), F(1.234567e-3));
  test(STR("0#012346"), loc, STR("{:.6Lf}"), F(1.234567e-2));
  test(STR("0#123457"), loc, STR("{:.6Lf}"), F(1.234567e-1));
  test(STR("1#234567"), loc, STR("{:.6Lf}"), F(1.234567e0));
  test(STR("1_2#345670"), loc, STR("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("12_3#456700"), loc, STR("{:.6Lf}"), F(1.234567e2));
    test(STR("1_23_4#567000"), loc, STR("{:.6Lf}"), F(1.234567e3));
    test(STR("12_34_5#670000"), loc, STR("{:.6Lf}"), F(1.234567e4));
    test(STR("123_45_6#700000"), loc, STR("{:.6Lf}"), F(1.234567e5));
    test(STR("1_234_56_7#000000"), loc, STR("{:.6Lf}"), F(1.234567e6));
    test(STR("12_345_67_0#000000"), loc, STR("{:.6Lf}"), F(1.234567e7));
    test(STR("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), loc, STR("{:.6Lf}"), F(1.234567e20));
  }
  test(STR("-0#000001"), loc, STR("{:.6Lf}"), F(-1.234567e-6));
  test(STR("-0#000012"), loc, STR("{:.6Lf}"), F(-1.234567e-5));
  test(STR("-0#000123"), loc, STR("{:.6Lf}"), F(-1.234567e-4));
  test(STR("-0#001235"), loc, STR("{:.6Lf}"), F(-1.234567e-3));
  test(STR("-0#012346"), loc, STR("{:.6Lf}"), F(-1.234567e-2));
  test(STR("-0#123457"), loc, STR("{:.6Lf}"), F(-1.234567e-1));
  test(STR("-1#234567"), loc, STR("{:.6Lf}"), F(-1.234567e0));
  test(STR("-1_2#345670"), loc, STR("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("-12_3#456700"), loc, STR("{:.6Lf}"), F(-1.234567e2));
    test(STR("-1_23_4#567000"), loc, STR("{:.6Lf}"), F(-1.234567e3));
    test(STR("-12_34_5#670000"), loc, STR("{:.6Lf}"), F(-1.234567e4));
    test(STR("-123_45_6#700000"), loc, STR("{:.6Lf}"), F(-1.234567e5));
    test(STR("-1_234_56_7#000000"), loc, STR("{:.6Lf}"), F(-1.234567e6));
    test(STR("-12_345_67_0#000000"), loc, STR("{:.6Lf}"), F(-1.234567e7));
    test(STR("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), loc, STR("{:.6Lf}"), F(-1.234567e20));
  }

  // *** Fill, align, zero padding ***
  if constexpr (sizeof(F) > sizeof(float)) {
    std::locale::global(en_US);
    test(STR("1,234.567000$$$"), STR("{:$<15.6Lf}"), F(1.234567e3));
    test(STR("$$$1,234.567000"), STR("{:$>15.6Lf}"), F(1.234567e3));
    test(STR("$1,234.567000$$"), STR("{:$^15.6Lf}"), F(1.234567e3));
    test(STR("0001,234.567000"), STR("{:015.6Lf}"), F(1.234567e3));
    test(STR("-1,234.567000$$$"), STR("{:$<16.6Lf}"), F(-1.234567e3));
    test(STR("$$$-1,234.567000"), STR("{:$>16.6Lf}"), F(-1.234567e3));
    test(STR("$-1,234.567000$$"), STR("{:$^16.6Lf}"), F(-1.234567e3));
    test(STR("-0001,234.567000"), STR("{:016.6Lf}"), F(-1.234567e3));

    std::locale::global(loc);
    test(STR("1_23_4#567000$$$"), STR("{:$<16.6Lf}"), F(1.234567e3));
    test(STR("$$$1_23_4#567000"), STR("{:$>16.6Lf}"), F(1.234567e3));
    test(STR("$1_23_4#567000$$"), STR("{:$^16.6Lf}"), F(1.234567e3));
    test(STR("0001_23_4#567000"), STR("{:016.6Lf}"), F(1.234567e3));
    test(STR("-1_23_4#567000$$$"), STR("{:$<17.6Lf}"), F(-1.234567e3));
    test(STR("$$$-1_23_4#567000"), STR("{:$>17.6Lf}"), F(-1.234567e3));
    test(STR("$-1_23_4#567000$$"), STR("{:$^17.6Lf}"), F(-1.234567e3));
    test(STR("-0001_23_4#567000"), STR("{:017.6Lf}"), F(-1.234567e3));

    test(STR("1,234.567000$$$"), en_US, STR("{:$<15.6Lf}"), F(1.234567e3));
    test(STR("$$$1,234.567000"), en_US, STR("{:$>15.6Lf}"), F(1.234567e3));
    test(STR("$1,234.567000$$"), en_US, STR("{:$^15.6Lf}"), F(1.234567e3));
    test(STR("0001,234.567000"), en_US, STR("{:015.6Lf}"), F(1.234567e3));
    test(STR("-1,234.567000$$$"), en_US, STR("{:$<16.6Lf}"), F(-1.234567e3));
    test(STR("$$$-1,234.567000"), en_US, STR("{:$>16.6Lf}"), F(-1.234567e3));
    test(STR("$-1,234.567000$$"), en_US, STR("{:$^16.6Lf}"), F(-1.234567e3));
    test(STR("-0001,234.567000"), en_US, STR("{:016.6Lf}"), F(-1.234567e3));

    std::locale::global(en_US);
    test(STR("1_23_4#567000$$$"), loc, STR("{:$<16.6Lf}"), F(1.234567e3));
    test(STR("$$$1_23_4#567000"), loc, STR("{:$>16.6Lf}"), F(1.234567e3));
    test(STR("$1_23_4#567000$$"), loc, STR("{:$^16.6Lf}"), F(1.234567e3));
    test(STR("0001_23_4#567000"), loc, STR("{:016.6Lf}"), F(1.234567e3));
    test(STR("-1_23_4#567000$$$"), loc, STR("{:$<17.6Lf}"), F(-1.234567e3));
    test(STR("$$$-1_23_4#567000"), loc, STR("{:$>17.6Lf}"), F(-1.234567e3));
    test(STR("$-1_23_4#567000$$"), loc, STR("{:$^17.6Lf}"), F(-1.234567e3));
    test(STR("-0001_23_4#567000"), loc, STR("{:017.6Lf}"), F(-1.234567e3));
  }
}

template <class F, class CharT>
void test_floating_point_fixed_upper_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(STR("0.000001"), STR("{:.6Lf}"), F(1.234567e-6));
  test(STR("0.000012"), STR("{:.6Lf}"), F(1.234567e-5));
  test(STR("0.000123"), STR("{:.6Lf}"), F(1.234567e-4));
  test(STR("0.001235"), STR("{:.6Lf}"), F(1.234567e-3));
  test(STR("0.012346"), STR("{:.6Lf}"), F(1.234567e-2));
  test(STR("0.123457"), STR("{:.6Lf}"), F(1.234567e-1));
  test(STR("1.234567"), STR("{:.6Lf}"), F(1.234567e0));
  test(STR("12.345670"), STR("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("123.456700"), STR("{:.6Lf}"), F(1.234567e2));
    test(STR("1,234.567000"), STR("{:.6Lf}"), F(1.234567e3));
    test(STR("12,345.670000"), STR("{:.6Lf}"), F(1.234567e4));
    test(STR("123,456.700000"), STR("{:.6Lf}"), F(1.234567e5));
    test(STR("1,234,567.000000"), STR("{:.6Lf}"), F(1.234567e6));
    test(STR("12,345,670.000000"), STR("{:.6Lf}"), F(1.234567e7));
    test(STR("123,456,700,000,000,000,000.000000"), STR("{:.6Lf}"), F(1.234567e20));
  }
  test(STR("-0.000001"), STR("{:.6Lf}"), F(-1.234567e-6));
  test(STR("-0.000012"), STR("{:.6Lf}"), F(-1.234567e-5));
  test(STR("-0.000123"), STR("{:.6Lf}"), F(-1.234567e-4));
  test(STR("-0.001235"), STR("{:.6Lf}"), F(-1.234567e-3));
  test(STR("-0.012346"), STR("{:.6Lf}"), F(-1.234567e-2));
  test(STR("-0.123457"), STR("{:.6Lf}"), F(-1.234567e-1));
  test(STR("-1.234567"), STR("{:.6Lf}"), F(-1.234567e0));
  test(STR("-12.345670"), STR("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("-123.456700"), STR("{:.6Lf}"), F(-1.234567e2));
    test(STR("-1,234.567000"), STR("{:.6Lf}"), F(-1.234567e3));
    test(STR("-12,345.670000"), STR("{:.6Lf}"), F(-1.234567e4));
    test(STR("-123,456.700000"), STR("{:.6Lf}"), F(-1.234567e5));
    test(STR("-1,234,567.000000"), STR("{:.6Lf}"), F(-1.234567e6));
    test(STR("-12,345,670.000000"), STR("{:.6Lf}"), F(-1.234567e7));
    test(STR("-123,456,700,000,000,000,000.000000"), STR("{:.6Lf}"), F(-1.234567e20));
  }

  std::locale::global(loc);
  test(STR("0#000001"), STR("{:.6Lf}"), F(1.234567e-6));
  test(STR("0#000012"), STR("{:.6Lf}"), F(1.234567e-5));
  test(STR("0#000123"), STR("{:.6Lf}"), F(1.234567e-4));
  test(STR("0#001235"), STR("{:.6Lf}"), F(1.234567e-3));
  test(STR("0#012346"), STR("{:.6Lf}"), F(1.234567e-2));
  test(STR("0#123457"), STR("{:.6Lf}"), F(1.234567e-1));
  test(STR("1#234567"), STR("{:.6Lf}"), F(1.234567e0));
  test(STR("1_2#345670"), STR("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("12_3#456700"), STR("{:.6Lf}"), F(1.234567e2));
    test(STR("1_23_4#567000"), STR("{:.6Lf}"), F(1.234567e3));
    test(STR("12_34_5#670000"), STR("{:.6Lf}"), F(1.234567e4));
    test(STR("123_45_6#700000"), STR("{:.6Lf}"), F(1.234567e5));
    test(STR("1_234_56_7#000000"), STR("{:.6Lf}"), F(1.234567e6));
    test(STR("12_345_67_0#000000"), STR("{:.6Lf}"), F(1.234567e7));
    test(STR("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), STR("{:.6Lf}"), F(1.234567e20));
  }
  test(STR("-0#000001"), STR("{:.6Lf}"), F(-1.234567e-6));
  test(STR("-0#000012"), STR("{:.6Lf}"), F(-1.234567e-5));
  test(STR("-0#000123"), STR("{:.6Lf}"), F(-1.234567e-4));
  test(STR("-0#001235"), STR("{:.6Lf}"), F(-1.234567e-3));
  test(STR("-0#012346"), STR("{:.6Lf}"), F(-1.234567e-2));
  test(STR("-0#123457"), STR("{:.6Lf}"), F(-1.234567e-1));
  test(STR("-1#234567"), STR("{:.6Lf}"), F(-1.234567e0));
  test(STR("-1_2#345670"), STR("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("-12_3#456700"), STR("{:.6Lf}"), F(-1.234567e2));
    test(STR("-1_23_4#567000"), STR("{:.6Lf}"), F(-1.234567e3));
    test(STR("-12_34_5#670000"), STR("{:.6Lf}"), F(-1.234567e4));
    test(STR("-123_45_6#700000"), STR("{:.6Lf}"), F(-1.234567e5));
    test(STR("-1_234_56_7#000000"), STR("{:.6Lf}"), F(-1.234567e6));
    test(STR("-12_345_67_0#000000"), STR("{:.6Lf}"), F(-1.234567e7));
    test(STR("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), STR("{:.6Lf}"), F(-1.234567e20));
  }

  test(STR("0.000001"), en_US, STR("{:.6Lf}"), F(1.234567e-6));
  test(STR("0.000012"), en_US, STR("{:.6Lf}"), F(1.234567e-5));
  test(STR("0.000123"), en_US, STR("{:.6Lf}"), F(1.234567e-4));
  test(STR("0.001235"), en_US, STR("{:.6Lf}"), F(1.234567e-3));
  test(STR("0.012346"), en_US, STR("{:.6Lf}"), F(1.234567e-2));
  test(STR("0.123457"), en_US, STR("{:.6Lf}"), F(1.234567e-1));
  test(STR("1.234567"), en_US, STR("{:.6Lf}"), F(1.234567e0));
  test(STR("12.345670"), en_US, STR("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("123.456700"), en_US, STR("{:.6Lf}"), F(1.234567e2));
    test(STR("1,234.567000"), en_US, STR("{:.6Lf}"), F(1.234567e3));
    test(STR("12,345.670000"), en_US, STR("{:.6Lf}"), F(1.234567e4));
    test(STR("123,456.700000"), en_US, STR("{:.6Lf}"), F(1.234567e5));
    test(STR("1,234,567.000000"), en_US, STR("{:.6Lf}"), F(1.234567e6));
    test(STR("12,345,670.000000"), en_US, STR("{:.6Lf}"), F(1.234567e7));
    test(STR("123,456,700,000,000,000,000.000000"), en_US, STR("{:.6Lf}"), F(1.234567e20));
  }
  test(STR("-0.000001"), en_US, STR("{:.6Lf}"), F(-1.234567e-6));
  test(STR("-0.000012"), en_US, STR("{:.6Lf}"), F(-1.234567e-5));
  test(STR("-0.000123"), en_US, STR("{:.6Lf}"), F(-1.234567e-4));
  test(STR("-0.001235"), en_US, STR("{:.6Lf}"), F(-1.234567e-3));
  test(STR("-0.012346"), en_US, STR("{:.6Lf}"), F(-1.234567e-2));
  test(STR("-0.123457"), en_US, STR("{:.6Lf}"), F(-1.234567e-1));
  test(STR("-1.234567"), en_US, STR("{:.6Lf}"), F(-1.234567e0));
  test(STR("-12.345670"), en_US, STR("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("-123.456700"), en_US, STR("{:.6Lf}"), F(-1.234567e2));
    test(STR("-1,234.567000"), en_US, STR("{:.6Lf}"), F(-1.234567e3));
    test(STR("-12,345.670000"), en_US, STR("{:.6Lf}"), F(-1.234567e4));
    test(STR("-123,456.700000"), en_US, STR("{:.6Lf}"), F(-1.234567e5));
    test(STR("-1,234,567.000000"), en_US, STR("{:.6Lf}"), F(-1.234567e6));
    test(STR("-12,345,670.000000"), en_US, STR("{:.6Lf}"), F(-1.234567e7));
    test(STR("-123,456,700,000,000,000,000.000000"), en_US, STR("{:.6Lf}"), F(-1.234567e20));
  }

  std::locale::global(en_US);
  test(STR("0#000001"), loc, STR("{:.6Lf}"), F(1.234567e-6));
  test(STR("0#000012"), loc, STR("{:.6Lf}"), F(1.234567e-5));
  test(STR("0#000123"), loc, STR("{:.6Lf}"), F(1.234567e-4));
  test(STR("0#001235"), loc, STR("{:.6Lf}"), F(1.234567e-3));
  test(STR("0#012346"), loc, STR("{:.6Lf}"), F(1.234567e-2));
  test(STR("0#123457"), loc, STR("{:.6Lf}"), F(1.234567e-1));
  test(STR("1#234567"), loc, STR("{:.6Lf}"), F(1.234567e0));
  test(STR("1_2#345670"), loc, STR("{:.6Lf}"), F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("12_3#456700"), loc, STR("{:.6Lf}"), F(1.234567e2));
    test(STR("1_23_4#567000"), loc, STR("{:.6Lf}"), F(1.234567e3));
    test(STR("12_34_5#670000"), loc, STR("{:.6Lf}"), F(1.234567e4));
    test(STR("123_45_6#700000"), loc, STR("{:.6Lf}"), F(1.234567e5));
    test(STR("1_234_56_7#000000"), loc, STR("{:.6Lf}"), F(1.234567e6));
    test(STR("12_345_67_0#000000"), loc, STR("{:.6Lf}"), F(1.234567e7));
    test(STR("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), loc, STR("{:.6Lf}"), F(1.234567e20));
  }
  test(STR("-0#000001"), loc, STR("{:.6Lf}"), F(-1.234567e-6));
  test(STR("-0#000012"), loc, STR("{:.6Lf}"), F(-1.234567e-5));
  test(STR("-0#000123"), loc, STR("{:.6Lf}"), F(-1.234567e-4));
  test(STR("-0#001235"), loc, STR("{:.6Lf}"), F(-1.234567e-3));
  test(STR("-0#012346"), loc, STR("{:.6Lf}"), F(-1.234567e-2));
  test(STR("-0#123457"), loc, STR("{:.6Lf}"), F(-1.234567e-1));
  test(STR("-1#234567"), loc, STR("{:.6Lf}"), F(-1.234567e0));
  test(STR("-1_2#345670"), loc, STR("{:.6Lf}"), F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("-12_3#456700"), loc, STR("{:.6Lf}"), F(-1.234567e2));
    test(STR("-1_23_4#567000"), loc, STR("{:.6Lf}"), F(-1.234567e3));
    test(STR("-12_34_5#670000"), loc, STR("{:.6Lf}"), F(-1.234567e4));
    test(STR("-123_45_6#700000"), loc, STR("{:.6Lf}"), F(-1.234567e5));
    test(STR("-1_234_56_7#000000"), loc, STR("{:.6Lf}"), F(-1.234567e6));
    test(STR("-12_345_67_0#000000"), loc, STR("{:.6Lf}"), F(-1.234567e7));
    test(STR("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000"), loc, STR("{:.6Lf}"), F(-1.234567e20));
  }

  // *** Fill, align, zero padding ***
  if constexpr (sizeof(F) > sizeof(float)) {
    std::locale::global(en_US);
    test(STR("1,234.567000$$$"), STR("{:$<15.6Lf}"), F(1.234567e3));
    test(STR("$$$1,234.567000"), STR("{:$>15.6Lf}"), F(1.234567e3));
    test(STR("$1,234.567000$$"), STR("{:$^15.6Lf}"), F(1.234567e3));
    test(STR("0001,234.567000"), STR("{:015.6Lf}"), F(1.234567e3));
    test(STR("-1,234.567000$$$"), STR("{:$<16.6Lf}"), F(-1.234567e3));
    test(STR("$$$-1,234.567000"), STR("{:$>16.6Lf}"), F(-1.234567e3));
    test(STR("$-1,234.567000$$"), STR("{:$^16.6Lf}"), F(-1.234567e3));
    test(STR("-0001,234.567000"), STR("{:016.6Lf}"), F(-1.234567e3));

    std::locale::global(loc);
    test(STR("1_23_4#567000$$$"), STR("{:$<16.6Lf}"), F(1.234567e3));
    test(STR("$$$1_23_4#567000"), STR("{:$>16.6Lf}"), F(1.234567e3));
    test(STR("$1_23_4#567000$$"), STR("{:$^16.6Lf}"), F(1.234567e3));
    test(STR("0001_23_4#567000"), STR("{:016.6Lf}"), F(1.234567e3));
    test(STR("-1_23_4#567000$$$"), STR("{:$<17.6Lf}"), F(-1.234567e3));
    test(STR("$$$-1_23_4#567000"), STR("{:$>17.6Lf}"), F(-1.234567e3));
    test(STR("$-1_23_4#567000$$"), STR("{:$^17.6Lf}"), F(-1.234567e3));
    test(STR("-0001_23_4#567000"), STR("{:017.6Lf}"), F(-1.234567e3));

    test(STR("1,234.567000$$$"), en_US, STR("{:$<15.6Lf}"), F(1.234567e3));
    test(STR("$$$1,234.567000"), en_US, STR("{:$>15.6Lf}"), F(1.234567e3));
    test(STR("$1,234.567000$$"), en_US, STR("{:$^15.6Lf}"), F(1.234567e3));
    test(STR("0001,234.567000"), en_US, STR("{:015.6Lf}"), F(1.234567e3));
    test(STR("-1,234.567000$$$"), en_US, STR("{:$<16.6Lf}"), F(-1.234567e3));
    test(STR("$$$-1,234.567000"), en_US, STR("{:$>16.6Lf}"), F(-1.234567e3));
    test(STR("$-1,234.567000$$"), en_US, STR("{:$^16.6Lf}"), F(-1.234567e3));
    test(STR("-0001,234.567000"), en_US, STR("{:016.6Lf}"), F(-1.234567e3));

    std::locale::global(en_US);
    test(STR("1_23_4#567000$$$"), loc, STR("{:$<16.6Lf}"), F(1.234567e3));
    test(STR("$$$1_23_4#567000"), loc, STR("{:$>16.6Lf}"), F(1.234567e3));
    test(STR("$1_23_4#567000$$"), loc, STR("{:$^16.6Lf}"), F(1.234567e3));
    test(STR("0001_23_4#567000"), loc, STR("{:016.6Lf}"), F(1.234567e3));
    test(STR("-1_23_4#567000$$$"), loc, STR("{:$<17.6Lf}"), F(-1.234567e3));
    test(STR("$$$-1_23_4#567000"), loc, STR("{:$>17.6Lf}"), F(-1.234567e3));
    test(STR("$-1_23_4#567000$$"), loc, STR("{:$^17.6Lf}"), F(-1.234567e3));
    test(STR("-0001_23_4#567000"), loc, STR("{:017.6Lf}"), F(-1.234567e3));
  }
}

template <class F, class CharT>
void test_floating_point_general_lower_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(STR("1.23457e-06"), STR("{:.6Lg}"), F(1.234567e-6));
  test(STR("1.23457e-05"), STR("{:.6Lg}"), F(1.234567e-5));
  test(STR("0.000123457"), STR("{:.6Lg}"), F(1.234567e-4));
  test(STR("0.00123457"), STR("{:.6Lg}"), F(1.234567e-3));
  test(STR("0.0123457"), STR("{:.6Lg}"), F(1.234567e-2));
  test(STR("0.123457"), STR("{:.6Lg}"), F(1.234567e-1));
  test(STR("1.23457"), STR("{:.6Lg}"), F(1.234567e0));
  test(STR("12.3457"), STR("{:.6Lg}"), F(1.234567e1));
  test(STR("123.457"), STR("{:.6Lg}"), F(1.234567e2));
  test(STR("1,234.57"), STR("{:.6Lg}"), F(1.234567e3));
  test(STR("12,345.7"), STR("{:.6Lg}"), F(1.234567e4));
  test(STR("123,457"), STR("{:.6Lg}"), F(1.234567e5));
  test(STR("1.23457e+06"), STR("{:.6Lg}"), F(1.234567e6));
  test(STR("1.23457e+07"), STR("{:.6Lg}"), F(1.234567e7));
  test(STR("-1.23457e-06"), STR("{:.6Lg}"), F(-1.234567e-6));
  test(STR("-1.23457e-05"), STR("{:.6Lg}"), F(-1.234567e-5));
  test(STR("-0.000123457"), STR("{:.6Lg}"), F(-1.234567e-4));
  test(STR("-0.00123457"), STR("{:.6Lg}"), F(-1.234567e-3));
  test(STR("-0.0123457"), STR("{:.6Lg}"), F(-1.234567e-2));
  test(STR("-0.123457"), STR("{:.6Lg}"), F(-1.234567e-1));
  test(STR("-1.23457"), STR("{:.6Lg}"), F(-1.234567e0));
  test(STR("-12.3457"), STR("{:.6Lg}"), F(-1.234567e1));
  test(STR("-123.457"), STR("{:.6Lg}"), F(-1.234567e2));
  test(STR("-1,234.57"), STR("{:.6Lg}"), F(-1.234567e3));
  test(STR("-12,345.7"), STR("{:.6Lg}"), F(-1.234567e4));
  test(STR("-123,457"), STR("{:.6Lg}"), F(-1.234567e5));
  test(STR("-1.23457e+06"), STR("{:.6Lg}"), F(-1.234567e6));
  test(STR("-1.23457e+07"), STR("{:.6Lg}"), F(-1.234567e7));

  std::locale::global(loc);
  test(STR("1#23457e-06"), STR("{:.6Lg}"), F(1.234567e-6));
  test(STR("1#23457e-05"), STR("{:.6Lg}"), F(1.234567e-5));
  test(STR("0#000123457"), STR("{:.6Lg}"), F(1.234567e-4));
  test(STR("0#00123457"), STR("{:.6Lg}"), F(1.234567e-3));
  test(STR("0#0123457"), STR("{:.6Lg}"), F(1.234567e-2));
  test(STR("0#123457"), STR("{:.6Lg}"), F(1.234567e-1));
  test(STR("1#23457"), STR("{:.6Lg}"), F(1.234567e0));
  test(STR("1_2#3457"), STR("{:.6Lg}"), F(1.234567e1));
  test(STR("12_3#457"), STR("{:.6Lg}"), F(1.234567e2));
  test(STR("1_23_4#57"), STR("{:.6Lg}"), F(1.234567e3));
  test(STR("12_34_5#7"), STR("{:.6Lg}"), F(1.234567e4));
  test(STR("123_45_7"), STR("{:.6Lg}"), F(1.234567e5));
  test(STR("1#23457e+06"), STR("{:.6Lg}"), F(1.234567e6));
  test(STR("1#23457e+07"), STR("{:.6Lg}"), F(1.234567e7));
  test(STR("-1#23457e-06"), STR("{:.6Lg}"), F(-1.234567e-6));
  test(STR("-1#23457e-05"), STR("{:.6Lg}"), F(-1.234567e-5));
  test(STR("-0#000123457"), STR("{:.6Lg}"), F(-1.234567e-4));
  test(STR("-0#00123457"), STR("{:.6Lg}"), F(-1.234567e-3));
  test(STR("-0#0123457"), STR("{:.6Lg}"), F(-1.234567e-2));
  test(STR("-0#123457"), STR("{:.6Lg}"), F(-1.234567e-1));
  test(STR("-1#23457"), STR("{:.6Lg}"), F(-1.234567e0));
  test(STR("-1_2#3457"), STR("{:.6Lg}"), F(-1.234567e1));
  test(STR("-12_3#457"), STR("{:.6Lg}"), F(-1.234567e2));
  test(STR("-1_23_4#57"), STR("{:.6Lg}"), F(-1.234567e3));
  test(STR("-12_34_5#7"), STR("{:.6Lg}"), F(-1.234567e4));
  test(STR("-123_45_7"), STR("{:.6Lg}"), F(-1.234567e5));
  test(STR("-1#23457e+06"), STR("{:.6Lg}"), F(-1.234567e6));
  test(STR("-1#23457e+07"), STR("{:.6Lg}"), F(-1.234567e7));

  test(STR("1.23457e-06"), en_US, STR("{:.6Lg}"), F(1.234567e-6));
  test(STR("1.23457e-05"), en_US, STR("{:.6Lg}"), F(1.234567e-5));
  test(STR("0.000123457"), en_US, STR("{:.6Lg}"), F(1.234567e-4));
  test(STR("0.00123457"), en_US, STR("{:.6Lg}"), F(1.234567e-3));
  test(STR("0.0123457"), en_US, STR("{:.6Lg}"), F(1.234567e-2));
  test(STR("0.123457"), en_US, STR("{:.6Lg}"), F(1.234567e-1));
  test(STR("1.23457"), en_US, STR("{:.6Lg}"), F(1.234567e0));
  test(STR("12.3457"), en_US, STR("{:.6Lg}"), F(1.234567e1));
  test(STR("123.457"), en_US, STR("{:.6Lg}"), F(1.234567e2));
  test(STR("1,234.57"), en_US, STR("{:.6Lg}"), F(1.234567e3));
  test(STR("12,345.7"), en_US, STR("{:.6Lg}"), F(1.234567e4));
  test(STR("123,457"), en_US, STR("{:.6Lg}"), F(1.234567e5));
  test(STR("1.23457e+06"), en_US, STR("{:.6Lg}"), F(1.234567e6));
  test(STR("1.23457e+07"), en_US, STR("{:.6Lg}"), F(1.234567e7));
  test(STR("-1.23457e-06"), en_US, STR("{:.6Lg}"), F(-1.234567e-6));
  test(STR("-1.23457e-05"), en_US, STR("{:.6Lg}"), F(-1.234567e-5));
  test(STR("-0.000123457"), en_US, STR("{:.6Lg}"), F(-1.234567e-4));
  test(STR("-0.00123457"), en_US, STR("{:.6Lg}"), F(-1.234567e-3));
  test(STR("-0.0123457"), en_US, STR("{:.6Lg}"), F(-1.234567e-2));
  test(STR("-0.123457"), en_US, STR("{:.6Lg}"), F(-1.234567e-1));
  test(STR("-1.23457"), en_US, STR("{:.6Lg}"), F(-1.234567e0));
  test(STR("-12.3457"), en_US, STR("{:.6Lg}"), F(-1.234567e1));
  test(STR("-123.457"), en_US, STR("{:.6Lg}"), F(-1.234567e2));
  test(STR("-1,234.57"), en_US, STR("{:.6Lg}"), F(-1.234567e3));
  test(STR("-12,345.7"), en_US, STR("{:.6Lg}"), F(-1.234567e4));
  test(STR("-123,457"), en_US, STR("{:.6Lg}"), F(-1.234567e5));
  test(STR("-1.23457e+06"), en_US, STR("{:.6Lg}"), F(-1.234567e6));
  test(STR("-1.23457e+07"), en_US, STR("{:.6Lg}"), F(-1.234567e7));

  std::locale::global(en_US);
  test(STR("1#23457e-06"), loc, STR("{:.6Lg}"), F(1.234567e-6));
  test(STR("1#23457e-05"), loc, STR("{:.6Lg}"), F(1.234567e-5));
  test(STR("0#000123457"), loc, STR("{:.6Lg}"), F(1.234567e-4));
  test(STR("0#00123457"), loc, STR("{:.6Lg}"), F(1.234567e-3));
  test(STR("0#0123457"), loc, STR("{:.6Lg}"), F(1.234567e-2));
  test(STR("0#123457"), loc, STR("{:.6Lg}"), F(1.234567e-1));
  test(STR("1#23457"), loc, STR("{:.6Lg}"), F(1.234567e0));
  test(STR("1_2#3457"), loc, STR("{:.6Lg}"), F(1.234567e1));
  test(STR("12_3#457"), loc, STR("{:.6Lg}"), F(1.234567e2));
  test(STR("1_23_4#57"), loc, STR("{:.6Lg}"), F(1.234567e3));
  test(STR("12_34_5#7"), loc, STR("{:.6Lg}"), F(1.234567e4));
  test(STR("123_45_7"), loc, STR("{:.6Lg}"), F(1.234567e5));
  test(STR("1#23457e+06"), loc, STR("{:.6Lg}"), F(1.234567e6));
  test(STR("1#23457e+07"), loc, STR("{:.6Lg}"), F(1.234567e7));
  test(STR("-1#23457e-06"), loc, STR("{:.6Lg}"), F(-1.234567e-6));
  test(STR("-1#23457e-05"), loc, STR("{:.6Lg}"), F(-1.234567e-5));
  test(STR("-0#000123457"), loc, STR("{:.6Lg}"), F(-1.234567e-4));
  test(STR("-0#00123457"), loc, STR("{:.6Lg}"), F(-1.234567e-3));
  test(STR("-0#0123457"), loc, STR("{:.6Lg}"), F(-1.234567e-2));
  test(STR("-0#123457"), loc, STR("{:.6Lg}"), F(-1.234567e-1));
  test(STR("-1#23457"), loc, STR("{:.6Lg}"), F(-1.234567e0));
  test(STR("-1_2#3457"), loc, STR("{:.6Lg}"), F(-1.234567e1));
  test(STR("-12_3#457"), loc, STR("{:.6Lg}"), F(-1.234567e2));
  test(STR("-1_23_4#57"), loc, STR("{:.6Lg}"), F(-1.234567e3));
  test(STR("-12_34_5#7"), loc, STR("{:.6Lg}"), F(-1.234567e4));
  test(STR("-123_45_7"), loc, STR("{:.6Lg}"), F(-1.234567e5));
  test(STR("-1#23457e+06"), loc, STR("{:.6Lg}"), F(-1.234567e6));
  test(STR("-1#23457e+07"), loc, STR("{:.6Lg}"), F(-1.234567e7));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(STR("1,234.57$$$"), STR("{:$<11.6Lg}"), F(1.234567e3));
  test(STR("$$$1,234.57"), STR("{:$>11.6Lg}"), F(1.234567e3));
  test(STR("$1,234.57$$"), STR("{:$^11.6Lg}"), F(1.234567e3));
  test(STR("0001,234.57"), STR("{:011.6Lg}"), F(1.234567e3));
  test(STR("-1,234.57$$$"), STR("{:$<12.6Lg}"), F(-1.234567e3));
  test(STR("$$$-1,234.57"), STR("{:$>12.6Lg}"), F(-1.234567e3));
  test(STR("$-1,234.57$$"), STR("{:$^12.6Lg}"), F(-1.234567e3));
  test(STR("-0001,234.57"), STR("{:012.6Lg}"), F(-1.234567e3));

  std::locale::global(loc);
  test(STR("1_23_4#57$$$"), STR("{:$<12.6Lg}"), F(1.234567e3));
  test(STR("$$$1_23_4#57"), STR("{:$>12.6Lg}"), F(1.234567e3));
  test(STR("$1_23_4#57$$"), STR("{:$^12.6Lg}"), F(1.234567e3));
  test(STR("0001_23_4#57"), STR("{:012.6Lg}"), F(1.234567e3));
  test(STR("-1_23_4#57$$$"), STR("{:$<13.6Lg}"), F(-1.234567e3));
  test(STR("$$$-1_23_4#57"), STR("{:$>13.6Lg}"), F(-1.234567e3));
  test(STR("$-1_23_4#57$$"), STR("{:$^13.6Lg}"), F(-1.234567e3));
  test(STR("-0001_23_4#57"), STR("{:013.6Lg}"), F(-1.234567e3));

  test(STR("1,234.57$$$"), en_US, STR("{:$<11.6Lg}"), F(1.234567e3));
  test(STR("$$$1,234.57"), en_US, STR("{:$>11.6Lg}"), F(1.234567e3));
  test(STR("$1,234.57$$"), en_US, STR("{:$^11.6Lg}"), F(1.234567e3));
  test(STR("0001,234.57"), en_US, STR("{:011.6Lg}"), F(1.234567e3));
  test(STR("-1,234.57$$$"), en_US, STR("{:$<12.6Lg}"), F(-1.234567e3));
  test(STR("$$$-1,234.57"), en_US, STR("{:$>12.6Lg}"), F(-1.234567e3));
  test(STR("$-1,234.57$$"), en_US, STR("{:$^12.6Lg}"), F(-1.234567e3));
  test(STR("-0001,234.57"), en_US, STR("{:012.6Lg}"), F(-1.234567e3));

  std::locale::global(en_US);
  test(STR("1_23_4#57$$$"), loc, STR("{:$<12.6Lg}"), F(1.234567e3));
  test(STR("$$$1_23_4#57"), loc, STR("{:$>12.6Lg}"), F(1.234567e3));
  test(STR("$1_23_4#57$$"), loc, STR("{:$^12.6Lg}"), F(1.234567e3));
  test(STR("0001_23_4#57"), loc, STR("{:012.6Lg}"), F(1.234567e3));
  test(STR("-1_23_4#57$$$"), loc, STR("{:$<13.6Lg}"), F(-1.234567e3));
  test(STR("$$$-1_23_4#57"), loc, STR("{:$>13.6Lg}"), F(-1.234567e3));
  test(STR("$-1_23_4#57$$"), loc, STR("{:$^13.6Lg}"), F(-1.234567e3));
  test(STR("-0001_23_4#57"), loc, STR("{:013.6Lg}"), F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_general_upper_case() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(STR("1.23457E-06"), STR("{:.6LG}"), F(1.234567e-6));
  test(STR("1.23457E-05"), STR("{:.6LG}"), F(1.234567e-5));
  test(STR("0.000123457"), STR("{:.6LG}"), F(1.234567e-4));
  test(STR("0.00123457"), STR("{:.6LG}"), F(1.234567e-3));
  test(STR("0.0123457"), STR("{:.6LG}"), F(1.234567e-2));
  test(STR("0.123457"), STR("{:.6LG}"), F(1.234567e-1));
  test(STR("1.23457"), STR("{:.6LG}"), F(1.234567e0));
  test(STR("12.3457"), STR("{:.6LG}"), F(1.234567e1));
  test(STR("123.457"), STR("{:.6LG}"), F(1.234567e2));
  test(STR("1,234.57"), STR("{:.6LG}"), F(1.234567e3));
  test(STR("12,345.7"), STR("{:.6LG}"), F(1.234567e4));
  test(STR("123,457"), STR("{:.6LG}"), F(1.234567e5));
  test(STR("1.23457E+06"), STR("{:.6LG}"), F(1.234567e6));
  test(STR("1.23457E+07"), STR("{:.6LG}"), F(1.234567e7));
  test(STR("-1.23457E-06"), STR("{:.6LG}"), F(-1.234567e-6));
  test(STR("-1.23457E-05"), STR("{:.6LG}"), F(-1.234567e-5));
  test(STR("-0.000123457"), STR("{:.6LG}"), F(-1.234567e-4));
  test(STR("-0.00123457"), STR("{:.6LG}"), F(-1.234567e-3));
  test(STR("-0.0123457"), STR("{:.6LG}"), F(-1.234567e-2));
  test(STR("-0.123457"), STR("{:.6LG}"), F(-1.234567e-1));
  test(STR("-1.23457"), STR("{:.6LG}"), F(-1.234567e0));
  test(STR("-12.3457"), STR("{:.6LG}"), F(-1.234567e1));
  test(STR("-123.457"), STR("{:.6LG}"), F(-1.234567e2));
  test(STR("-1,234.57"), STR("{:.6LG}"), F(-1.234567e3));
  test(STR("-12,345.7"), STR("{:.6LG}"), F(-1.234567e4));
  test(STR("-123,457"), STR("{:.6LG}"), F(-1.234567e5));
  test(STR("-1.23457E+06"), STR("{:.6LG}"), F(-1.234567e6));
  test(STR("-1.23457E+07"), STR("{:.6LG}"), F(-1.234567e7));

  std::locale::global(loc);
  test(STR("1#23457E-06"), STR("{:.6LG}"), F(1.234567e-6));
  test(STR("1#23457E-05"), STR("{:.6LG}"), F(1.234567e-5));
  test(STR("0#000123457"), STR("{:.6LG}"), F(1.234567e-4));
  test(STR("0#00123457"), STR("{:.6LG}"), F(1.234567e-3));
  test(STR("0#0123457"), STR("{:.6LG}"), F(1.234567e-2));
  test(STR("0#123457"), STR("{:.6LG}"), F(1.234567e-1));
  test(STR("1#23457"), STR("{:.6LG}"), F(1.234567e0));
  test(STR("1_2#3457"), STR("{:.6LG}"), F(1.234567e1));
  test(STR("12_3#457"), STR("{:.6LG}"), F(1.234567e2));
  test(STR("1_23_4#57"), STR("{:.6LG}"), F(1.234567e3));
  test(STR("12_34_5#7"), STR("{:.6LG}"), F(1.234567e4));
  test(STR("123_45_7"), STR("{:.6LG}"), F(1.234567e5));
  test(STR("1#23457E+06"), STR("{:.6LG}"), F(1.234567e6));
  test(STR("1#23457E+07"), STR("{:.6LG}"), F(1.234567e7));
  test(STR("-1#23457E-06"), STR("{:.6LG}"), F(-1.234567e-6));
  test(STR("-1#23457E-05"), STR("{:.6LG}"), F(-1.234567e-5));
  test(STR("-0#000123457"), STR("{:.6LG}"), F(-1.234567e-4));
  test(STR("-0#00123457"), STR("{:.6LG}"), F(-1.234567e-3));
  test(STR("-0#0123457"), STR("{:.6LG}"), F(-1.234567e-2));
  test(STR("-0#123457"), STR("{:.6LG}"), F(-1.234567e-1));
  test(STR("-1#23457"), STR("{:.6LG}"), F(-1.234567e0));
  test(STR("-1_2#3457"), STR("{:.6LG}"), F(-1.234567e1));
  test(STR("-12_3#457"), STR("{:.6LG}"), F(-1.234567e2));
  test(STR("-1_23_4#57"), STR("{:.6LG}"), F(-1.234567e3));
  test(STR("-12_34_5#7"), STR("{:.6LG}"), F(-1.234567e4));
  test(STR("-123_45_7"), STR("{:.6LG}"), F(-1.234567e5));
  test(STR("-1#23457E+06"), STR("{:.6LG}"), F(-1.234567e6));
  test(STR("-1#23457E+07"), STR("{:.6LG}"), F(-1.234567e7));

  test(STR("1.23457E-06"), en_US, STR("{:.6LG}"), F(1.234567e-6));
  test(STR("1.23457E-05"), en_US, STR("{:.6LG}"), F(1.234567e-5));
  test(STR("0.000123457"), en_US, STR("{:.6LG}"), F(1.234567e-4));
  test(STR("0.00123457"), en_US, STR("{:.6LG}"), F(1.234567e-3));
  test(STR("0.0123457"), en_US, STR("{:.6LG}"), F(1.234567e-2));
  test(STR("0.123457"), en_US, STR("{:.6LG}"), F(1.234567e-1));
  test(STR("1.23457"), en_US, STR("{:.6LG}"), F(1.234567e0));
  test(STR("12.3457"), en_US, STR("{:.6LG}"), F(1.234567e1));
  test(STR("123.457"), en_US, STR("{:.6LG}"), F(1.234567e2));
  test(STR("1,234.57"), en_US, STR("{:.6LG}"), F(1.234567e3));
  test(STR("12,345.7"), en_US, STR("{:.6LG}"), F(1.234567e4));
  test(STR("123,457"), en_US, STR("{:.6LG}"), F(1.234567e5));
  test(STR("1.23457E+06"), en_US, STR("{:.6LG}"), F(1.234567e6));
  test(STR("1.23457E+07"), en_US, STR("{:.6LG}"), F(1.234567e7));
  test(STR("-1.23457E-06"), en_US, STR("{:.6LG}"), F(-1.234567e-6));
  test(STR("-1.23457E-05"), en_US, STR("{:.6LG}"), F(-1.234567e-5));
  test(STR("-0.000123457"), en_US, STR("{:.6LG}"), F(-1.234567e-4));
  test(STR("-0.00123457"), en_US, STR("{:.6LG}"), F(-1.234567e-3));
  test(STR("-0.0123457"), en_US, STR("{:.6LG}"), F(-1.234567e-2));
  test(STR("-0.123457"), en_US, STR("{:.6LG}"), F(-1.234567e-1));
  test(STR("-1.23457"), en_US, STR("{:.6LG}"), F(-1.234567e0));
  test(STR("-12.3457"), en_US, STR("{:.6LG}"), F(-1.234567e1));
  test(STR("-123.457"), en_US, STR("{:.6LG}"), F(-1.234567e2));
  test(STR("-1,234.57"), en_US, STR("{:.6LG}"), F(-1.234567e3));
  test(STR("-12,345.7"), en_US, STR("{:.6LG}"), F(-1.234567e4));
  test(STR("-123,457"), en_US, STR("{:.6LG}"), F(-1.234567e5));
  test(STR("-1.23457E+06"), en_US, STR("{:.6LG}"), F(-1.234567e6));
  test(STR("-1.23457E+07"), en_US, STR("{:.6LG}"), F(-1.234567e7));

  std::locale::global(en_US);
  test(STR("1#23457E-06"), loc, STR("{:.6LG}"), F(1.234567e-6));
  test(STR("1#23457E-05"), loc, STR("{:.6LG}"), F(1.234567e-5));
  test(STR("0#000123457"), loc, STR("{:.6LG}"), F(1.234567e-4));
  test(STR("0#00123457"), loc, STR("{:.6LG}"), F(1.234567e-3));
  test(STR("0#0123457"), loc, STR("{:.6LG}"), F(1.234567e-2));
  test(STR("0#123457"), loc, STR("{:.6LG}"), F(1.234567e-1));
  test(STR("1#23457"), loc, STR("{:.6LG}"), F(1.234567e0));
  test(STR("1_2#3457"), loc, STR("{:.6LG}"), F(1.234567e1));
  test(STR("12_3#457"), loc, STR("{:.6LG}"), F(1.234567e2));
  test(STR("1_23_4#57"), loc, STR("{:.6LG}"), F(1.234567e3));
  test(STR("12_34_5#7"), loc, STR("{:.6LG}"), F(1.234567e4));
  test(STR("123_45_7"), loc, STR("{:.6LG}"), F(1.234567e5));
  test(STR("1#23457E+06"), loc, STR("{:.6LG}"), F(1.234567e6));
  test(STR("1#23457E+07"), loc, STR("{:.6LG}"), F(1.234567e7));
  test(STR("-1#23457E-06"), loc, STR("{:.6LG}"), F(-1.234567e-6));
  test(STR("-1#23457E-05"), loc, STR("{:.6LG}"), F(-1.234567e-5));
  test(STR("-0#000123457"), loc, STR("{:.6LG}"), F(-1.234567e-4));
  test(STR("-0#00123457"), loc, STR("{:.6LG}"), F(-1.234567e-3));
  test(STR("-0#0123457"), loc, STR("{:.6LG}"), F(-1.234567e-2));
  test(STR("-0#123457"), loc, STR("{:.6LG}"), F(-1.234567e-1));
  test(STR("-1#23457"), loc, STR("{:.6LG}"), F(-1.234567e0));
  test(STR("-1_2#3457"), loc, STR("{:.6LG}"), F(-1.234567e1));
  test(STR("-12_3#457"), loc, STR("{:.6LG}"), F(-1.234567e2));
  test(STR("-1_23_4#57"), loc, STR("{:.6LG}"), F(-1.234567e3));
  test(STR("-12_34_5#7"), loc, STR("{:.6LG}"), F(-1.234567e4));
  test(STR("-123_45_7"), loc, STR("{:.6LG}"), F(-1.234567e5));
  test(STR("-1#23457E+06"), loc, STR("{:.6LG}"), F(-1.234567e6));
  test(STR("-1#23457E+07"), loc, STR("{:.6LG}"), F(-1.234567e7));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(STR("1,234.57$$$"), STR("{:$<11.6LG}"), F(1.234567e3));
  test(STR("$$$1,234.57"), STR("{:$>11.6LG}"), F(1.234567e3));
  test(STR("$1,234.57$$"), STR("{:$^11.6LG}"), F(1.234567e3));
  test(STR("0001,234.57"), STR("{:011.6LG}"), F(1.234567e3));
  test(STR("-1,234.57$$$"), STR("{:$<12.6LG}"), F(-1.234567e3));
  test(STR("$$$-1,234.57"), STR("{:$>12.6LG}"), F(-1.234567e3));
  test(STR("$-1,234.57$$"), STR("{:$^12.6LG}"), F(-1.234567e3));
  test(STR("-0001,234.57"), STR("{:012.6LG}"), F(-1.234567e3));

  std::locale::global(loc);
  test(STR("1_23_4#57$$$"), STR("{:$<12.6LG}"), F(1.234567e3));
  test(STR("$$$1_23_4#57"), STR("{:$>12.6LG}"), F(1.234567e3));
  test(STR("$1_23_4#57$$"), STR("{:$^12.6LG}"), F(1.234567e3));
  test(STR("0001_23_4#57"), STR("{:012.6LG}"), F(1.234567e3));
  test(STR("-1_23_4#57$$$"), STR("{:$<13.6LG}"), F(-1.234567e3));
  test(STR("$$$-1_23_4#57"), STR("{:$>13.6LG}"), F(-1.234567e3));
  test(STR("$-1_23_4#57$$"), STR("{:$^13.6LG}"), F(-1.234567e3));
  test(STR("-0001_23_4#57"), STR("{:013.6LG}"), F(-1.234567e3));

  test(STR("1,234.57$$$"), en_US, STR("{:$<11.6LG}"), F(1.234567e3));
  test(STR("$$$1,234.57"), en_US, STR("{:$>11.6LG}"), F(1.234567e3));
  test(STR("$1,234.57$$"), en_US, STR("{:$^11.6LG}"), F(1.234567e3));
  test(STR("0001,234.57"), en_US, STR("{:011.6LG}"), F(1.234567e3));
  test(STR("-1,234.57$$$"), en_US, STR("{:$<12.6LG}"), F(-1.234567e3));
  test(STR("$$$-1,234.57"), en_US, STR("{:$>12.6LG}"), F(-1.234567e3));
  test(STR("$-1,234.57$$"), en_US, STR("{:$^12.6LG}"), F(-1.234567e3));
  test(STR("-0001,234.57"), en_US, STR("{:012.6LG}"), F(-1.234567e3));

  std::locale::global(en_US);
  test(STR("1_23_4#57$$$"), loc, STR("{:$<12.6LG}"), F(1.234567e3));
  test(STR("$$$1_23_4#57"), loc, STR("{:$>12.6LG}"), F(1.234567e3));
  test(STR("$1_23_4#57$$"), loc, STR("{:$^12.6LG}"), F(1.234567e3));
  test(STR("0001_23_4#57"), loc, STR("{:012.6LG}"), F(1.234567e3));
  test(STR("-1_23_4#57$$$"), loc, STR("{:$<13.6LG}"), F(-1.234567e3));
  test(STR("$$$-1_23_4#57"), loc, STR("{:$>13.6LG}"), F(-1.234567e3));
  test(STR("$-1_23_4#57$$"), loc, STR("{:$^13.6LG}"), F(-1.234567e3));
  test(STR("-0001_23_4#57"), loc, STR("{:013.6LG}"), F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_default() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(STR("1.234567e-06"), STR("{:L}"), F(1.234567e-6));
  test(STR("1.234567e-05"), STR("{:L}"), F(1.234567e-5));
  test(STR("0.0001234567"), STR("{:L}"), F(1.234567e-4));
  test(STR("0.001234567"), STR("{:L}"), F(1.234567e-3));
  test(STR("0.01234567"), STR("{:L}"), F(1.234567e-2));
  test(STR("0.1234567"), STR("{:L}"), F(1.234567e-1));
  test(STR("1.234567"), STR("{:L}"), F(1.234567e0));
  test(STR("12.34567"), STR("{:L}"), F(1.234567e1));
  test(STR("123.4567"), STR("{:L}"), F(1.234567e2));
  test(STR("1,234.567"), STR("{:L}"), F(1.234567e3));
  test(STR("12,345.67"), STR("{:L}"), F(1.234567e4));
  test(STR("123,456.7"), STR("{:L}"), F(1.234567e5));
  test(STR("1,234,567"), STR("{:L}"), F(1.234567e6));
  test(STR("12,345,670"), STR("{:L}"), F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("123,456,700"), STR("{:L}"), F(1.234567e8));
    test(STR("1,234,567,000"), STR("{:L}"), F(1.234567e9));
    test(STR("12,345,670,000"), STR("{:L}"), F(1.234567e10));
    test(STR("123,456,700,000"), STR("{:L}"), F(1.234567e11));
    test(STR("1.234567e+12"), STR("{:L}"), F(1.234567e12));
    test(STR("1.234567e+13"), STR("{:L}"), F(1.234567e13));
  }
  test(STR("-1.234567e-06"), STR("{:L}"), F(-1.234567e-6));
  test(STR("-1.234567e-05"), STR("{:L}"), F(-1.234567e-5));
  test(STR("-0.0001234567"), STR("{:L}"), F(-1.234567e-4));
  test(STR("-0.001234567"), STR("{:L}"), F(-1.234567e-3));
  test(STR("-0.01234567"), STR("{:L}"), F(-1.234567e-2));
  test(STR("-0.1234567"), STR("{:L}"), F(-1.234567e-1));
  test(STR("-1.234567"), STR("{:L}"), F(-1.234567e0));
  test(STR("-12.34567"), STR("{:L}"), F(-1.234567e1));
  test(STR("-123.4567"), STR("{:L}"), F(-1.234567e2));
  test(STR("-1,234.567"), STR("{:L}"), F(-1.234567e3));
  test(STR("-12,345.67"), STR("{:L}"), F(-1.234567e4));
  test(STR("-123,456.7"), STR("{:L}"), F(-1.234567e5));
  test(STR("-1,234,567"), STR("{:L}"), F(-1.234567e6));
  test(STR("-12,345,670"), STR("{:L}"), F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("-123,456,700"), STR("{:L}"), F(-1.234567e8));
    test(STR("-1,234,567,000"), STR("{:L}"), F(-1.234567e9));
    test(STR("-12,345,670,000"), STR("{:L}"), F(-1.234567e10));
    test(STR("-123,456,700,000"), STR("{:L}"), F(-1.234567e11));
    test(STR("-1.234567e+12"), STR("{:L}"), F(-1.234567e12));
    test(STR("-1.234567e+13"), STR("{:L}"), F(-1.234567e13));
  }

  std::locale::global(loc);
  test(STR("1#234567e-06"), STR("{:L}"), F(1.234567e-6));
  test(STR("1#234567e-05"), STR("{:L}"), F(1.234567e-5));
  test(STR("0#0001234567"), STR("{:L}"), F(1.234567e-4));
  test(STR("0#001234567"), STR("{:L}"), F(1.234567e-3));
  test(STR("0#01234567"), STR("{:L}"), F(1.234567e-2));
  test(STR("0#1234567"), STR("{:L}"), F(1.234567e-1));
  test(STR("1#234567"), STR("{:L}"), F(1.234567e0));
  test(STR("1_2#34567"), STR("{:L}"), F(1.234567e1));
  test(STR("12_3#4567"), STR("{:L}"), F(1.234567e2));
  test(STR("1_23_4#567"), STR("{:L}"), F(1.234567e3));
  test(STR("12_34_5#67"), STR("{:L}"), F(1.234567e4));
  test(STR("123_45_6#7"), STR("{:L}"), F(1.234567e5));
  test(STR("1_234_56_7"), STR("{:L}"), F(1.234567e6));
  test(STR("12_345_67_0"), STR("{:L}"), F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("1_23_456_70_0"), STR("{:L}"), F(1.234567e8));
    test(STR("1_2_34_567_00_0"), STR("{:L}"), F(1.234567e9));
    test(STR("1_2_3_45_670_00_0"), STR("{:L}"), F(1.234567e10));
    test(STR("1_2_3_4_56_700_00_0"), STR("{:L}"), F(1.234567e11));
    test(STR("1#234567e+12"), STR("{:L}"), F(1.234567e12));
    test(STR("1#234567e+13"), STR("{:L}"), F(1.234567e13));
  }
  test(STR("-1#234567e-06"), STR("{:L}"), F(-1.234567e-6));
  test(STR("-1#234567e-05"), STR("{:L}"), F(-1.234567e-5));
  test(STR("-0#0001234567"), STR("{:L}"), F(-1.234567e-4));
  test(STR("-0#001234567"), STR("{:L}"), F(-1.234567e-3));
  test(STR("-0#01234567"), STR("{:L}"), F(-1.234567e-2));
  test(STR("-0#1234567"), STR("{:L}"), F(-1.234567e-1));
  test(STR("-1#234567"), STR("{:L}"), F(-1.234567e0));
  test(STR("-1_2#34567"), STR("{:L}"), F(-1.234567e1));
  test(STR("-12_3#4567"), STR("{:L}"), F(-1.234567e2));
  test(STR("-1_23_4#567"), STR("{:L}"), F(-1.234567e3));
  test(STR("-12_34_5#67"), STR("{:L}"), F(-1.234567e4));
  test(STR("-123_45_6#7"), STR("{:L}"), F(-1.234567e5));
  test(STR("-1_234_56_7"), STR("{:L}"), F(-1.234567e6));
  test(STR("-12_345_67_0"), STR("{:L}"), F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("-1_23_456_70_0"), STR("{:L}"), F(-1.234567e8));
    test(STR("-1_2_34_567_00_0"), STR("{:L}"), F(-1.234567e9));
    test(STR("-1_2_3_45_670_00_0"), STR("{:L}"), F(-1.234567e10));
    test(STR("-1_2_3_4_56_700_00_0"), STR("{:L}"), F(-1.234567e11));
    test(STR("-1#234567e+12"), STR("{:L}"), F(-1.234567e12));
    test(STR("-1#234567e+13"), STR("{:L}"), F(-1.234567e13));
  }

  test(STR("1.234567e-06"), en_US, STR("{:L}"), F(1.234567e-6));
  test(STR("1.234567e-05"), en_US, STR("{:L}"), F(1.234567e-5));
  test(STR("0.0001234567"), en_US, STR("{:L}"), F(1.234567e-4));
  test(STR("0.001234567"), en_US, STR("{:L}"), F(1.234567e-3));
  test(STR("0.01234567"), en_US, STR("{:L}"), F(1.234567e-2));
  test(STR("0.1234567"), en_US, STR("{:L}"), F(1.234567e-1));
  test(STR("1.234567"), en_US, STR("{:L}"), F(1.234567e0));
  test(STR("12.34567"), en_US, STR("{:L}"), F(1.234567e1));
  test(STR("123.4567"), en_US, STR("{:L}"), F(1.234567e2));
  test(STR("1,234.567"), en_US, STR("{:L}"), F(1.234567e3));
  test(STR("12,345.67"), en_US, STR("{:L}"), F(1.234567e4));
  test(STR("123,456.7"), en_US, STR("{:L}"), F(1.234567e5));
  test(STR("1,234,567"), en_US, STR("{:L}"), F(1.234567e6));
  test(STR("12,345,670"), en_US, STR("{:L}"), F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("123,456,700"), en_US, STR("{:L}"), F(1.234567e8));
    test(STR("1,234,567,000"), en_US, STR("{:L}"), F(1.234567e9));
    test(STR("12,345,670,000"), en_US, STR("{:L}"), F(1.234567e10));
    test(STR("123,456,700,000"), en_US, STR("{:L}"), F(1.234567e11));
    test(STR("1.234567e+12"), en_US, STR("{:L}"), F(1.234567e12));
    test(STR("1.234567e+13"), en_US, STR("{:L}"), F(1.234567e13));
  }
  test(STR("-1.234567e-06"), en_US, STR("{:L}"), F(-1.234567e-6));
  test(STR("-1.234567e-05"), en_US, STR("{:L}"), F(-1.234567e-5));
  test(STR("-0.0001234567"), en_US, STR("{:L}"), F(-1.234567e-4));
  test(STR("-0.001234567"), en_US, STR("{:L}"), F(-1.234567e-3));
  test(STR("-0.01234567"), en_US, STR("{:L}"), F(-1.234567e-2));
  test(STR("-0.1234567"), en_US, STR("{:L}"), F(-1.234567e-1));
  test(STR("-1.234567"), en_US, STR("{:L}"), F(-1.234567e0));
  test(STR("-12.34567"), en_US, STR("{:L}"), F(-1.234567e1));
  test(STR("-123.4567"), en_US, STR("{:L}"), F(-1.234567e2));
  test(STR("-1,234.567"), en_US, STR("{:L}"), F(-1.234567e3));
  test(STR("-12,345.67"), en_US, STR("{:L}"), F(-1.234567e4));
  test(STR("-123,456.7"), en_US, STR("{:L}"), F(-1.234567e5));
  test(STR("-1,234,567"), en_US, STR("{:L}"), F(-1.234567e6));
  test(STR("-12,345,670"), en_US, STR("{:L}"), F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("-123,456,700"), en_US, STR("{:L}"), F(-1.234567e8));
    test(STR("-1,234,567,000"), en_US, STR("{:L}"), F(-1.234567e9));
    test(STR("-12,345,670,000"), en_US, STR("{:L}"), F(-1.234567e10));
    test(STR("-123,456,700,000"), en_US, STR("{:L}"), F(-1.234567e11));
    test(STR("-1.234567e+12"), en_US, STR("{:L}"), F(-1.234567e12));
    test(STR("-1.234567e+13"), en_US, STR("{:L}"), F(-1.234567e13));
  }

  std::locale::global(en_US);
  test(STR("1#234567e-06"), loc, STR("{:L}"), F(1.234567e-6));
  test(STR("1#234567e-05"), loc, STR("{:L}"), F(1.234567e-5));
  test(STR("0#0001234567"), loc, STR("{:L}"), F(1.234567e-4));
  test(STR("0#001234567"), loc, STR("{:L}"), F(1.234567e-3));
  test(STR("0#01234567"), loc, STR("{:L}"), F(1.234567e-2));
  test(STR("0#1234567"), loc, STR("{:L}"), F(1.234567e-1));
  test(STR("1#234567"), loc, STR("{:L}"), F(1.234567e0));
  test(STR("1_2#34567"), loc, STR("{:L}"), F(1.234567e1));
  test(STR("12_3#4567"), loc, STR("{:L}"), F(1.234567e2));
  test(STR("1_23_4#567"), loc, STR("{:L}"), F(1.234567e3));
  test(STR("12_34_5#67"), loc, STR("{:L}"), F(1.234567e4));
  test(STR("123_45_6#7"), loc, STR("{:L}"), F(1.234567e5));
  test(STR("1_234_56_7"), loc, STR("{:L}"), F(1.234567e6));
  test(STR("12_345_67_0"), loc, STR("{:L}"), F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("1_23_456_70_0"), loc, STR("{:L}"), F(1.234567e8));
    test(STR("1_2_34_567_00_0"), loc, STR("{:L}"), F(1.234567e9));
    test(STR("1_2_3_45_670_00_0"), loc, STR("{:L}"), F(1.234567e10));
    test(STR("1_2_3_4_56_700_00_0"), loc, STR("{:L}"), F(1.234567e11));
    test(STR("1#234567e+12"), loc, STR("{:L}"), F(1.234567e12));
    test(STR("1#234567e+13"), loc, STR("{:L}"), F(1.234567e13));
  }
  test(STR("-1#234567e-06"), loc, STR("{:L}"), F(-1.234567e-6));
  test(STR("-1#234567e-05"), loc, STR("{:L}"), F(-1.234567e-5));
  test(STR("-0#0001234567"), loc, STR("{:L}"), F(-1.234567e-4));
  test(STR("-0#001234567"), loc, STR("{:L}"), F(-1.234567e-3));
  test(STR("-0#01234567"), loc, STR("{:L}"), F(-1.234567e-2));
  test(STR("-0#1234567"), loc, STR("{:L}"), F(-1.234567e-1));
  test(STR("-1#234567"), loc, STR("{:L}"), F(-1.234567e0));
  test(STR("-1_2#34567"), loc, STR("{:L}"), F(-1.234567e1));
  test(STR("-12_3#4567"), loc, STR("{:L}"), F(-1.234567e2));
  test(STR("-1_23_4#567"), loc, STR("{:L}"), F(-1.234567e3));
  test(STR("-12_34_5#67"), loc, STR("{:L}"), F(-1.234567e4));
  test(STR("-123_45_6#7"), loc, STR("{:L}"), F(-1.234567e5));
  test(STR("-1_234_56_7"), loc, STR("{:L}"), F(-1.234567e6));
  test(STR("-12_345_67_0"), loc, STR("{:L}"), F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test(STR("-1_23_456_70_0"), loc, STR("{:L}"), F(-1.234567e8));
    test(STR("-1_2_34_567_00_0"), loc, STR("{:L}"), F(-1.234567e9));
    test(STR("-1_2_3_45_670_00_0"), loc, STR("{:L}"), F(-1.234567e10));
    test(STR("-1_2_3_4_56_700_00_0"), loc, STR("{:L}"), F(-1.234567e11));
    test(STR("-1#234567e+12"), loc, STR("{:L}"), F(-1.234567e12));
    test(STR("-1#234567e+13"), loc, STR("{:L}"), F(-1.234567e13));
  }

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(STR("1,234.567$$$"), STR("{:$<12L}"), F(1.234567e3));
  test(STR("$$$1,234.567"), STR("{:$>12L}"), F(1.234567e3));
  test(STR("$1,234.567$$"), STR("{:$^12L}"), F(1.234567e3));
  test(STR("0001,234.567"), STR("{:012L}"), F(1.234567e3));
  test(STR("-1,234.567$$$"), STR("{:$<13L}"), F(-1.234567e3));
  test(STR("$$$-1,234.567"), STR("{:$>13L}"), F(-1.234567e3));
  test(STR("$-1,234.567$$"), STR("{:$^13L}"), F(-1.234567e3));
  test(STR("-0001,234.567"), STR("{:013L}"), F(-1.234567e3));

  std::locale::global(loc);
  test(STR("1_23_4#567$$$"), STR("{:$<13L}"), F(1.234567e3));
  test(STR("$$$1_23_4#567"), STR("{:$>13L}"), F(1.234567e3));
  test(STR("$1_23_4#567$$"), STR("{:$^13L}"), F(1.234567e3));
  test(STR("0001_23_4#567"), STR("{:013L}"), F(1.234567e3));
  test(STR("-1_23_4#567$$$"), STR("{:$<14L}"), F(-1.234567e3));
  test(STR("$$$-1_23_4#567"), STR("{:$>14L}"), F(-1.234567e3));
  test(STR("$-1_23_4#567$$"), STR("{:$^14L}"), F(-1.234567e3));
  test(STR("-0001_23_4#567"), STR("{:014L}"), F(-1.234567e3));

  test(STR("1,234.567$$$"), en_US, STR("{:$<12L}"), F(1.234567e3));
  test(STR("$$$1,234.567"), en_US, STR("{:$>12L}"), F(1.234567e3));
  test(STR("$1,234.567$$"), en_US, STR("{:$^12L}"), F(1.234567e3));
  test(STR("0001,234.567"), en_US, STR("{:012L}"), F(1.234567e3));
  test(STR("-1,234.567$$$"), en_US, STR("{:$<13L}"), F(-1.234567e3));
  test(STR("$$$-1,234.567"), en_US, STR("{:$>13L}"), F(-1.234567e3));
  test(STR("$-1,234.567$$"), en_US, STR("{:$^13L}"), F(-1.234567e3));
  test(STR("-0001,234.567"), en_US, STR("{:013L}"), F(-1.234567e3));

  std::locale::global(en_US);
  test(STR("1_23_4#567$$$"), loc, STR("{:$<13L}"), F(1.234567e3));
  test(STR("$$$1_23_4#567"), loc, STR("{:$>13L}"), F(1.234567e3));
  test(STR("$1_23_4#567$$"), loc, STR("{:$^13L}"), F(1.234567e3));
  test(STR("0001_23_4#567"), loc, STR("{:013L}"), F(1.234567e3));
  test(STR("-1_23_4#567$$$"), loc, STR("{:$<14L}"), F(-1.234567e3));
  test(STR("$$$-1_23_4#567"), loc, STR("{:$>14L}"), F(-1.234567e3));
  test(STR("$-1_23_4#567$$"), loc, STR("{:$^14L}"), F(-1.234567e3));
  test(STR("-0001_23_4#567"), loc, STR("{:014L}"), F(-1.234567e3));
}

template <class F, class CharT>
void test_floating_point_default_precision() {
  std::locale loc = std::locale(std::locale(), new numpunct<CharT>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test(STR("1.23457e-06"), STR("{:.6L}"), F(1.234567e-6));
  test(STR("1.23457e-05"), STR("{:.6L}"), F(1.234567e-5));
  test(STR("0.000123457"), STR("{:.6L}"), F(1.234567e-4));
  test(STR("0.00123457"), STR("{:.6L}"), F(1.234567e-3));
  test(STR("0.0123457"), STR("{:.6L}"), F(1.234567e-2));
  test(STR("0.123457"), STR("{:.6L}"), F(1.234567e-1));
  test(STR("1.23457"), STR("{:.6L}"), F(1.234567e0));
  test(STR("12.3457"), STR("{:.6L}"), F(1.234567e1));
  test(STR("123.457"), STR("{:.6L}"), F(1.234567e2));
  test(STR("1,234.57"), STR("{:.6L}"), F(1.234567e3));
  test(STR("12,345.7"), STR("{:.6L}"), F(1.234567e4));
  test(STR("123,457"), STR("{:.6L}"), F(1.234567e5));
  test(STR("1.23457e+06"), STR("{:.6L}"), F(1.234567e6));
  test(STR("1.23457e+07"), STR("{:.6L}"), F(1.234567e7));
  test(STR("-1.23457e-06"), STR("{:.6L}"), F(-1.234567e-6));
  test(STR("-1.23457e-05"), STR("{:.6L}"), F(-1.234567e-5));
  test(STR("-0.000123457"), STR("{:.6L}"), F(-1.234567e-4));
  test(STR("-0.00123457"), STR("{:.6L}"), F(-1.234567e-3));
  test(STR("-0.0123457"), STR("{:.6L}"), F(-1.234567e-2));
  test(STR("-0.123457"), STR("{:.6L}"), F(-1.234567e-1));
  test(STR("-1.23457"), STR("{:.6L}"), F(-1.234567e0));
  test(STR("-12.3457"), STR("{:.6L}"), F(-1.234567e1));
  test(STR("-123.457"), STR("{:.6L}"), F(-1.234567e2));
  test(STR("-1,234.57"), STR("{:.6L}"), F(-1.234567e3));
  test(STR("-12,345.7"), STR("{:.6L}"), F(-1.234567e4));
  test(STR("-123,457"), STR("{:.6L}"), F(-1.234567e5));
  test(STR("-1.23457e+06"), STR("{:.6L}"), F(-1.234567e6));
  test(STR("-1.23457e+07"), STR("{:.6L}"), F(-1.234567e7));

  std::locale::global(loc);
  test(STR("1#23457e-06"), STR("{:.6L}"), F(1.234567e-6));
  test(STR("1#23457e-05"), STR("{:.6L}"), F(1.234567e-5));
  test(STR("0#000123457"), STR("{:.6L}"), F(1.234567e-4));
  test(STR("0#00123457"), STR("{:.6L}"), F(1.234567e-3));
  test(STR("0#0123457"), STR("{:.6L}"), F(1.234567e-2));
  test(STR("0#123457"), STR("{:.6L}"), F(1.234567e-1));
  test(STR("1#23457"), STR("{:.6L}"), F(1.234567e0));
  test(STR("1_2#3457"), STR("{:.6L}"), F(1.234567e1));
  test(STR("12_3#457"), STR("{:.6L}"), F(1.234567e2));
  test(STR("1_23_4#57"), STR("{:.6L}"), F(1.234567e3));
  test(STR("12_34_5#7"), STR("{:.6L}"), F(1.234567e4));
  test(STR("123_45_7"), STR("{:.6L}"), F(1.234567e5));
  test(STR("1#23457e+06"), STR("{:.6L}"), F(1.234567e6));
  test(STR("1#23457e+07"), STR("{:.6L}"), F(1.234567e7));
  test(STR("-1#23457e-06"), STR("{:.6L}"), F(-1.234567e-6));
  test(STR("-1#23457e-05"), STR("{:.6L}"), F(-1.234567e-5));
  test(STR("-0#000123457"), STR("{:.6L}"), F(-1.234567e-4));
  test(STR("-0#00123457"), STR("{:.6L}"), F(-1.234567e-3));
  test(STR("-0#0123457"), STR("{:.6L}"), F(-1.234567e-2));
  test(STR("-0#123457"), STR("{:.6L}"), F(-1.234567e-1));
  test(STR("-1#23457"), STR("{:.6L}"), F(-1.234567e0));
  test(STR("-1_2#3457"), STR("{:.6L}"), F(-1.234567e1));
  test(STR("-12_3#457"), STR("{:.6L}"), F(-1.234567e2));
  test(STR("-1_23_4#57"), STR("{:.6L}"), F(-1.234567e3));
  test(STR("-12_34_5#7"), STR("{:.6L}"), F(-1.234567e4));
  test(STR("-123_45_7"), STR("{:.6L}"), F(-1.234567e5));
  test(STR("-1#23457e+06"), STR("{:.6L}"), F(-1.234567e6));
  test(STR("-1#23457e+07"), STR("{:.6L}"), F(-1.234567e7));

  test(STR("1.23457e-06"), en_US, STR("{:.6L}"), F(1.234567e-6));
  test(STR("1.23457e-05"), en_US, STR("{:.6L}"), F(1.234567e-5));
  test(STR("0.000123457"), en_US, STR("{:.6L}"), F(1.234567e-4));
  test(STR("0.00123457"), en_US, STR("{:.6L}"), F(1.234567e-3));
  test(STR("0.0123457"), en_US, STR("{:.6L}"), F(1.234567e-2));
  test(STR("0.123457"), en_US, STR("{:.6L}"), F(1.234567e-1));
  test(STR("1.23457"), en_US, STR("{:.6L}"), F(1.234567e0));
  test(STR("12.3457"), en_US, STR("{:.6L}"), F(1.234567e1));
  test(STR("123.457"), en_US, STR("{:.6L}"), F(1.234567e2));
  test(STR("1,234.57"), en_US, STR("{:.6L}"), F(1.234567e3));
  test(STR("12,345.7"), en_US, STR("{:.6L}"), F(1.234567e4));
  test(STR("123,457"), en_US, STR("{:.6L}"), F(1.234567e5));
  test(STR("1.23457e+06"), en_US, STR("{:.6L}"), F(1.234567e6));
  test(STR("1.23457e+07"), en_US, STR("{:.6L}"), F(1.234567e7));
  test(STR("-1.23457e-06"), en_US, STR("{:.6L}"), F(-1.234567e-6));
  test(STR("-1.23457e-05"), en_US, STR("{:.6L}"), F(-1.234567e-5));
  test(STR("-0.000123457"), en_US, STR("{:.6L}"), F(-1.234567e-4));
  test(STR("-0.00123457"), en_US, STR("{:.6L}"), F(-1.234567e-3));
  test(STR("-0.0123457"), en_US, STR("{:.6L}"), F(-1.234567e-2));
  test(STR("-0.123457"), en_US, STR("{:.6L}"), F(-1.234567e-1));
  test(STR("-1.23457"), en_US, STR("{:.6L}"), F(-1.234567e0));
  test(STR("-12.3457"), en_US, STR("{:.6L}"), F(-1.234567e1));
  test(STR("-123.457"), en_US, STR("{:.6L}"), F(-1.234567e2));
  test(STR("-1,234.57"), en_US, STR("{:.6L}"), F(-1.234567e3));
  test(STR("-12,345.7"), en_US, STR("{:.6L}"), F(-1.234567e4));
  test(STR("-123,457"), en_US, STR("{:.6L}"), F(-1.234567e5));
  test(STR("-1.23457e+06"), en_US, STR("{:.6L}"), F(-1.234567e6));
  test(STR("-1.23457e+07"), en_US, STR("{:.6L}"), F(-1.234567e7));

  std::locale::global(en_US);
  test(STR("1#23457e-06"), loc, STR("{:.6L}"), F(1.234567e-6));
  test(STR("1#23457e-05"), loc, STR("{:.6L}"), F(1.234567e-5));
  test(STR("0#000123457"), loc, STR("{:.6L}"), F(1.234567e-4));
  test(STR("0#00123457"), loc, STR("{:.6L}"), F(1.234567e-3));
  test(STR("0#0123457"), loc, STR("{:.6L}"), F(1.234567e-2));
  test(STR("0#123457"), loc, STR("{:.6L}"), F(1.234567e-1));
  test(STR("1#23457"), loc, STR("{:.6L}"), F(1.234567e0));
  test(STR("1_2#3457"), loc, STR("{:.6L}"), F(1.234567e1));
  test(STR("12_3#457"), loc, STR("{:.6L}"), F(1.234567e2));
  test(STR("1_23_4#57"), loc, STR("{:.6L}"), F(1.234567e3));
  test(STR("12_34_5#7"), loc, STR("{:.6L}"), F(1.234567e4));
  test(STR("123_45_7"), loc, STR("{:.6L}"), F(1.234567e5));
  test(STR("1#23457e+06"), loc, STR("{:.6L}"), F(1.234567e6));
  test(STR("1#23457e+07"), loc, STR("{:.6L}"), F(1.234567e7));
  test(STR("-1#23457e-06"), loc, STR("{:.6L}"), F(-1.234567e-6));
  test(STR("-1#23457e-05"), loc, STR("{:.6L}"), F(-1.234567e-5));
  test(STR("-0#000123457"), loc, STR("{:.6L}"), F(-1.234567e-4));
  test(STR("-0#00123457"), loc, STR("{:.6L}"), F(-1.234567e-3));
  test(STR("-0#0123457"), loc, STR("{:.6L}"), F(-1.234567e-2));
  test(STR("-0#123457"), loc, STR("{:.6L}"), F(-1.234567e-1));
  test(STR("-1#23457"), loc, STR("{:.6L}"), F(-1.234567e0));
  test(STR("-1_2#3457"), loc, STR("{:.6L}"), F(-1.234567e1));
  test(STR("-12_3#457"), loc, STR("{:.6L}"), F(-1.234567e2));
  test(STR("-1_23_4#57"), loc, STR("{:.6L}"), F(-1.234567e3));
  test(STR("-12_34_5#7"), loc, STR("{:.6L}"), F(-1.234567e4));
  test(STR("-123_45_7"), loc, STR("{:.6L}"), F(-1.234567e5));
  test(STR("-1#23457e+06"), loc, STR("{:.6L}"), F(-1.234567e6));
  test(STR("-1#23457e+07"), loc, STR("{:.6L}"), F(-1.234567e7));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test(STR("1,234.57$$$"), STR("{:$<11.6L}"), F(1.234567e3));
  test(STR("$$$1,234.57"), STR("{:$>11.6L}"), F(1.234567e3));
  test(STR("$1,234.57$$"), STR("{:$^11.6L}"), F(1.234567e3));
  test(STR("0001,234.57"), STR("{:011.6L}"), F(1.234567e3));
  test(STR("-1,234.57$$$"), STR("{:$<12.6L}"), F(-1.234567e3));
  test(STR("$$$-1,234.57"), STR("{:$>12.6L}"), F(-1.234567e3));
  test(STR("$-1,234.57$$"), STR("{:$^12.6L}"), F(-1.234567e3));
  test(STR("-0001,234.57"), STR("{:012.6L}"), F(-1.234567e3));

  std::locale::global(loc);
  test(STR("1_23_4#57$$$"), STR("{:$<12.6L}"), F(1.234567e3));
  test(STR("$$$1_23_4#57"), STR("{:$>12.6L}"), F(1.234567e3));
  test(STR("$1_23_4#57$$"), STR("{:$^12.6L}"), F(1.234567e3));
  test(STR("0001_23_4#57"), STR("{:012.6L}"), F(1.234567e3));
  test(STR("-1_23_4#57$$$"), STR("{:$<13.6L}"), F(-1.234567e3));
  test(STR("$$$-1_23_4#57"), STR("{:$>13.6L}"), F(-1.234567e3));
  test(STR("$-1_23_4#57$$"), STR("{:$^13.6L}"), F(-1.234567e3));
  test(STR("-0001_23_4#57"), STR("{:013.6L}"), F(-1.234567e3));

  test(STR("1,234.57$$$"), en_US, STR("{:$<11.6L}"), F(1.234567e3));
  test(STR("$$$1,234.57"), en_US, STR("{:$>11.6L}"), F(1.234567e3));
  test(STR("$1,234.57$$"), en_US, STR("{:$^11.6L}"), F(1.234567e3));
  test(STR("0001,234.57"), en_US, STR("{:011.6L}"), F(1.234567e3));
  test(STR("-1,234.57$$$"), en_US, STR("{:$<12.6L}"), F(-1.234567e3));
  test(STR("$$$-1,234.57"), en_US, STR("{:$>12.6L}"), F(-1.234567e3));
  test(STR("$-1,234.57$$"), en_US, STR("{:$^12.6L}"), F(-1.234567e3));
  test(STR("-0001,234.57"), en_US, STR("{:012.6L}"), F(-1.234567e3));

  std::locale::global(en_US);
  test(STR("1_23_4#57$$$"), loc, STR("{:$<12.6L}"), F(1.234567e3));
  test(STR("$$$1_23_4#57"), loc, STR("{:$>12.6L}"), F(1.234567e3));
  test(STR("$1_23_4#57$$"), loc, STR("{:$^12.6L}"), F(1.234567e3));
  test(STR("0001_23_4#57"), loc, STR("{:012.6L}"), F(1.234567e3));
  test(STR("-1_23_4#57$$$"), loc, STR("{:$<13.6L}"), F(-1.234567e3));
  test(STR("$$$-1_23_4#57"), loc, STR("{:$>13.6L}"), F(-1.234567e3));
  test(STR("$-1_23_4#57$$"), loc, STR("{:$^13.6L}"), F(-1.234567e3));
  test(STR("-0001_23_4#57"), loc, STR("{:013.6L}"), F(-1.234567e3));
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
