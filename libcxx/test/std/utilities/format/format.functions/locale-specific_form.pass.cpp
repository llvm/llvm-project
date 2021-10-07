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
//    Out vformat_to(Out out, string_view fmt,
//                   format_args_t<type_identity_t<Out>, char> args);
//  template<class Out>
//    Out vformat_to(Out out, wstring_view fmt,
//                   format_args_t<type_identity_t<Out>, wchar_t> args);
//  template<class Out>
//    Out vformat_to(Out out, const locale& loc, string_view fmt,
//                   format_args_t<type_identity_t<Out>, char> args);
//  template<class Out>
//    Out vformat_to(Out out, const locale& loc, wstring_view fmt,
//                   format_args_t<type_identity_t<Out>, wchar_t> args);
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

#define STR(S) MAKE_STRING(CharT, S)

template <class CharT>
struct numpunct;

template <>
struct numpunct<char> : std::numpunct<char> {
  string_type do_truename() const override { return "yes"; }
  string_type do_falsename() const override { return "no"; }

  std::string do_grouping() const override { return "\1\2\3\2\1"; };
  char do_thousands_sep() const override { return '_'; }
};

template <>
struct numpunct<wchar_t> : std::numpunct<wchar_t> {
  string_type do_truename() const override { return L"yes"; }
  string_type do_falsename() const override { return L"no"; }

  std::string do_grouping() const override { return "\1\2\3\2\1"; };
  wchar_t do_thousands_sep() const override { return L'_'; }
};

template <class CharT, class... Args>
void test(std::basic_string<CharT> expected, std::basic_string<CharT> fmt,
          const Args&... args) {
  // *** format ***
  {
    std::basic_string<CharT> out = std::format(fmt, args...);
    if constexpr (std::same_as<CharT, char>)
      if (out != expected)
        std::cerr << "\nFormat string   " << fmt << "\nExpected output "
                  << expected << "\nActual output   " << out << '\n';
    assert(out == expected);
  }
  // *** vformat ***
  {
    std::basic_string<CharT> out = std::vformat(
        fmt, std::make_format_args<std::basic_format_context<
                 std::back_insert_iterator<std::basic_string<CharT>>, CharT>>(
                 args...));
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
    auto it = std::vformat_to(
        out.begin(), fmt,
        std::make_format_args<std::basic_format_context<
            typename std::basic_string<CharT>::iterator, CharT>>(args...));
    assert(it == out.end());
    assert(out == expected);
  }
  // *** format_to_n ***
  {
    std::basic_string<CharT> out;
    std::format_to_n_result result =
        std::format_to_n(std::back_inserter(out), 1000, fmt, args...);
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
void test(std::basic_string<CharT> expected, std::locale loc,
          std::basic_string<CharT> fmt, const Args&... args) {
  // *** format ***
  {
    std::basic_string<CharT> out = std::format(loc, fmt, args...);
    if constexpr (std::same_as<CharT, char>)
      if (out != expected)
        std::cerr << "\nFormat string   " << fmt << "\nExpected output "
                  << expected << "\nActual output   " << out << '\n';
    assert(out == expected);
  }
  // *** vformat ***
  {
    std::basic_string<CharT> out = std::vformat(
        loc, fmt,
        std::make_format_args<std::basic_format_context<
            std::back_insert_iterator<std::basic_string<CharT>>, CharT>>(
            args...));
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
    auto it = std::vformat_to(
        out.begin(), loc, fmt,
        std::make_format_args<std::basic_format_context<
            typename std::basic_string<CharT>::iterator, CharT>>(args...));
    assert(it == out.end());
    assert(out == expected);
  }
  // *** format_to_n ***
  {
    std::basic_string<CharT> out;
    std::format_to_n_result result =
        std::format_to_n(std::back_inserter(out), 1000, loc, fmt, args...);
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

template <class CharT>
void test() {
  test_integer<CharT>();
}

int main(int, char**) {
  test<char>();
  test<wchar_t>();

  return 0;
}
