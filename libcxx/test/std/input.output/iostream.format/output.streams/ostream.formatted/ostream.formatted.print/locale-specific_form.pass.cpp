//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-localization
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO PRINT Investigate see https://reviews.llvm.org/D156585
// UNSUPPORTED: no-filesystem

// XFAIL: availability-fp_to_chars-missing

// Bionic has minimal locale support, investigate this later.
// XFAIL: LIBCXX-ANDROID-FIXME

// REQUIRES: locale.en_US.UTF-8

// <format>

// This test checks the locale-specific form for these print functions:
// template<class... Args>
//   void print(ostream& os, format_string<Args...> fmt, Args&&... args);
// template<class... Args>
//   void println(ostream& os, format_string<Args...> fmt, Args&&... args);
//
// void vprint_unicode(ostream& os, string_view fmt, format_args args);
// void vprint_nonunicode(ostream& os, string_view fmt, format_args args);

#include <cassert>
#include <ostream>

#include "test_macros.h"
#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_format_string.h"
#include "assert_macros.h"
#include "concat_macros.h"

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

template <class... Args>
static void
test(std::stringstream& stream, std::string expected, test_format_string<char, Args...> fmt, Args&&... args) {
  // *** print ***
  {
    std::print(stream, fmt, std::forward<Args>(args)...);
    std::string out = stream.str();
    TEST_REQUIRE(out == expected,
                 TEST_WRITE_CONCATENATED(
                     "\nFormat string   ", fmt.get(), "\nExpected output ", expected, "\nActual output   ", out, '\n'));
  }
  // *** vprint_unicode ***
  {
    stream.str("");
    ;
    std::vprint_unicode(stream, fmt.get(), std::make_format_args(args...));
    std::string out = stream.str();
    TEST_REQUIRE(out == expected,
                 TEST_WRITE_CONCATENATED(
                     "\nFormat string   ", fmt.get(), "\nExpected output ", expected, "\nActual output   ", out, '\n'));
  }
  // *** vprint_nonunicode ***
  {
    stream.str("");
    ;
    std::vprint_nonunicode(stream, fmt.get(), std::make_format_args(args...));
    std::string out = stream.str();
    TEST_REQUIRE(out == expected,
                 TEST_WRITE_CONCATENATED(
                     "\nFormat string   ", fmt.get(), "\nExpected output ", expected, "\nActual output   ", out, '\n'));
  }
  // *** println ***
  {
    expected += '\n'; // Tested last since it changes the expected value.
    stream.str("");
    ;
    std::println(stream, fmt, std::forward<Args>(args)...);
    std::string out = stream.str();
    TEST_REQUIRE(out == expected,
                 TEST_WRITE_CONCATENATED(
                     "\nFormat string   ", fmt.get(), "\nExpected output ", expected, "\nActual output   ", out, '\n'));
  }
}

template <class... Args>
static void test(std::string expected, test_format_string<char, Args...> fmt, Args&&... args) {
  std::stringstream stream;
  test(stream, std::move(expected), fmt, std::forward<Args>(args)...);
}

template <class... Args>
static void test(std::string expected, std::locale loc, test_format_string<char, Args...> fmt, Args&&... args) {
  std::stringstream stream;
  stream.imbue(loc);
  test(stream, std::move(expected), fmt, std::forward<Args>(args)...);
}

#ifndef TEST_HAS_NO_UNICODE
struct numpunct_unicode : std::numpunct<char> {
  string_type do_truename() const override { return "gültig"; }
  string_type do_falsename() const override { return "ungültig"; }
};

#endif // TEST_HAS_NO_UNICODE

static void test_bool() {
  std::locale loc = std::locale(std::locale(), new numpunct<char>());

  std::locale::global(std::locale(LOCALE_en_US_UTF_8));
  assert(std::locale().name() == LOCALE_en_US_UTF_8);
  test("true", "{:L}", true);
  test("false", "{:L}", false);

  test("yes", loc, "{:L}", true);
  test("no", loc, "{:L}", false);

  std::locale::global(loc);
  test("yes", "{:L}", true);
  test("no", "{:L}", false);

  test("true", std::locale(LOCALE_en_US_UTF_8), "{:L}", true);
  test("false", std::locale(LOCALE_en_US_UTF_8), "{:L}", false);

#ifndef TEST_HAS_NO_UNICODE
  std::locale loc_unicode = std::locale(std::locale(), new numpunct_unicode());

  test("gültig", loc_unicode, "{:L}", true);
  test("ungültig", loc_unicode, "{:L}", false);

  test("gültig   ", loc_unicode, "{:9L}", true);
  test("gültig!!!", loc_unicode, "{:!<9L}", true);
  test("_gültig__", loc_unicode, "{:_^9L}", true);
  test("   gültig", loc_unicode, "{:>9L}", true);
#endif // TEST_HAS_NO_UNICODE
}

static void test_integer() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Decimal ***
  std::locale::global(en_US);
  test("0", "{:L}", 0);
  test("1", "{:L}", 1);
  test("10", "{:L}", 10);
  test("100", "{:L}", 100);
  test("1,000", "{:L}", 1'000);
  test("10,000", "{:L}", 10'000);
  test("100,000", "{:L}", 100'000);
  test("1,000,000", "{:L}", 1'000'000);
  test("10,000,000", "{:L}", 10'000'000);
  test("100,000,000", "{:L}", 100'000'000);
  test("1,000,000,000", "{:L}", 1'000'000'000);

  test("-1", "{:L}", -1);
  test("-10", "{:L}", -10);
  test("-100", "{:L}", -100);
  test("-1,000", "{:L}", -1'000);
  test("-10,000", "{:L}", -10'000);
  test("-100,000", "{:L}", -100'000);
  test("-1,000,000", "{:L}", -1'000'000);
  test("-10,000,000", "{:L}", -10'000'000);
  test("-100,000,000", "{:L}", -100'000'000);
  test("-1,000,000,000", "{:L}", -1'000'000'000);

  std::locale::global(loc);
  test("0", "{:L}", 0);
  test("1", "{:L}", 1);
  test("1_0", "{:L}", 10);
  test("10_0", "{:L}", 100);
  test("1_00_0", "{:L}", 1'000);
  test("10_00_0", "{:L}", 10'000);
  test("100_00_0", "{:L}", 100'000);
  test("1_000_00_0", "{:L}", 1'000'000);
  test("10_000_00_0", "{:L}", 10'000'000);
  test("1_00_000_00_0", "{:L}", 100'000'000);
  test("1_0_00_000_00_0", "{:L}", 1'000'000'000);

  test("-1", "{:L}", -1);
  test("-1_0", "{:L}", -10);
  test("-10_0", "{:L}", -100);
  test("-1_00_0", "{:L}", -1'000);
  test("-10_00_0", "{:L}", -10'000);
  test("-100_00_0", "{:L}", -100'000);
  test("-1_000_00_0", "{:L}", -1'000'000);
  test("-10_000_00_0", "{:L}", -10'000'000);
  test("-1_00_000_00_0", "{:L}", -100'000'000);
  test("-1_0_00_000_00_0", "{:L}", -1'000'000'000);

  test("0", en_US, "{:L}", 0);
  test("1", en_US, "{:L}", 1);
  test("10", en_US, "{:L}", 10);
  test("100", en_US, "{:L}", 100);
  test("1,000", en_US, "{:L}", 1'000);
  test("10,000", en_US, "{:L}", 10'000);
  test("100,000", en_US, "{:L}", 100'000);
  test("1,000,000", en_US, "{:L}", 1'000'000);
  test("10,000,000", en_US, "{:L}", 10'000'000);
  test("100,000,000", en_US, "{:L}", 100'000'000);
  test("1,000,000,000", en_US, "{:L}", 1'000'000'000);

  test("-1", en_US, "{:L}", -1);
  test("-10", en_US, "{:L}", -10);
  test("-100", en_US, "{:L}", -100);
  test("-1,000", en_US, "{:L}", -1'000);
  test("-10,000", en_US, "{:L}", -10'000);
  test("-100,000", en_US, "{:L}", -100'000);
  test("-1,000,000", en_US, "{:L}", -1'000'000);
  test("-10,000,000", en_US, "{:L}", -10'000'000);
  test("-100,000,000", en_US, "{:L}", -100'000'000);
  test("-1,000,000,000", en_US, "{:L}", -1'000'000'000);

  std::locale::global(en_US);
  test("0", loc, "{:L}", 0);
  test("1", loc, "{:L}", 1);
  test("1_0", loc, "{:L}", 10);
  test("10_0", loc, "{:L}", 100);
  test("1_00_0", loc, "{:L}", 1'000);
  test("10_00_0", loc, "{:L}", 10'000);
  test("100_00_0", loc, "{:L}", 100'000);
  test("1_000_00_0", loc, "{:L}", 1'000'000);
  test("10_000_00_0", loc, "{:L}", 10'000'000);
  test("1_00_000_00_0", loc, "{:L}", 100'000'000);
  test("1_0_00_000_00_0", loc, "{:L}", 1'000'000'000);

  test("-1", loc, "{:L}", -1);
  test("-1_0", loc, "{:L}", -10);
  test("-10_0", loc, "{:L}", -100);
  test("-1_00_0", loc, "{:L}", -1'000);
  test("-10_00_0", loc, "{:L}", -10'000);
  test("-100_00_0", loc, "{:L}", -100'000);
  test("-1_000_00_0", loc, "{:L}", -1'000'000);
  test("-10_000_00_0", loc, "{:L}", -10'000'000);
  test("-1_00_000_00_0", loc, "{:L}", -100'000'000);
  test("-1_0_00_000_00_0", loc, "{:L}", -1'000'000'000);

  // *** Binary ***
  std::locale::global(en_US);
  test("0", "{:Lb}", 0b0);
  test("1", "{:Lb}", 0b1);
  test("1,000,000,000", "{:Lb}", 0b1'000'000'000);

  test("0b0", "{:#Lb}", 0b0);
  test("0b1", "{:#Lb}", 0b1);
  test("0b1,000,000,000", "{:#Lb}", 0b1'000'000'000);

  test("-1", "{:LB}", -0b1);
  test("-1,000,000,000", "{:LB}", -0b1'000'000'000);

  test("-0B1", "{:#LB}", -0b1);
  test("-0B1,000,000,000", "{:#LB}", -0b1'000'000'000);

  std::locale::global(loc);
  test("0", "{:Lb}", 0b0);
  test("1", "{:Lb}", 0b1);
  test("1_0_00_000_00_0", "{:Lb}", 0b1'000'000'000);

  test("0b0", "{:#Lb}", 0b0);
  test("0b1", "{:#Lb}", 0b1);
  test("0b1_0_00_000_00_0", "{:#Lb}", 0b1'000'000'000);

  test("-1", "{:LB}", -0b1);
  test("-1_0_00_000_00_0", "{:LB}", -0b1'000'000'000);

  test("-0B1", "{:#LB}", -0b1);
  test("-0B1_0_00_000_00_0", "{:#LB}", -0b1'000'000'000);

  test("0", en_US, "{:Lb}", 0b0);
  test("1", en_US, "{:Lb}", 0b1);
  test("1,000,000,000", en_US, "{:Lb}", 0b1'000'000'000);

  test("0b0", en_US, "{:#Lb}", 0b0);
  test("0b1", en_US, "{:#Lb}", 0b1);
  test("0b1,000,000,000", en_US, "{:#Lb}", 0b1'000'000'000);

  test("-1", en_US, "{:LB}", -0b1);
  test("-1,000,000,000", en_US, "{:LB}", -0b1'000'000'000);

  test("-0B1", en_US, "{:#LB}", -0b1);
  test("-0B1,000,000,000", en_US, "{:#LB}", -0b1'000'000'000);

  std::locale::global(en_US);
  test("0", loc, "{:Lb}", 0b0);
  test("1", loc, "{:Lb}", 0b1);
  test("1_0_00_000_00_0", loc, "{:Lb}", 0b1'000'000'000);

  test("0b0", loc, "{:#Lb}", 0b0);
  test("0b1", loc, "{:#Lb}", 0b1);
  test("0b1_0_00_000_00_0", loc, "{:#Lb}", 0b1'000'000'000);

  test("-1", loc, "{:LB}", -0b1);
  test("-1_0_00_000_00_0", loc, "{:LB}", -0b1'000'000'000);

  test("-0B1", loc, "{:#LB}", -0b1);
  test("-0B1_0_00_000_00_0", loc, "{:#LB}", -0b1'000'000'000);

  // *** Octal ***
  std::locale::global(en_US);
  test("0", "{:Lo}", 00);
  test("1", "{:Lo}", 01);
  test("1,000,000,000", "{:Lo}", 01'000'000'000);

  test("0", "{:#Lo}", 00);
  test("01", "{:#Lo}", 01);
  test("01,000,000,000", "{:#Lo}", 01'000'000'000);

  test("-1", "{:Lo}", -01);
  test("-1,000,000,000", "{:Lo}", -01'000'000'000);

  test("-01", "{:#Lo}", -01);
  test("-01,000,000,000", "{:#Lo}", -01'000'000'000);

  std::locale::global(loc);
  test("0", "{:Lo}", 00);
  test("1", "{:Lo}", 01);
  test("1_0_00_000_00_0", "{:Lo}", 01'000'000'000);

  test("0", "{:#Lo}", 00);
  test("01", "{:#Lo}", 01);
  test("01_0_00_000_00_0", "{:#Lo}", 01'000'000'000);

  test("-1", "{:Lo}", -01);
  test("-1_0_00_000_00_0", "{:Lo}", -01'000'000'000);

  test("-01", "{:#Lo}", -01);
  test("-01_0_00_000_00_0", "{:#Lo}", -01'000'000'000);

  test("0", en_US, "{:Lo}", 00);
  test("1", en_US, "{:Lo}", 01);
  test("1,000,000,000", en_US, "{:Lo}", 01'000'000'000);

  test("0", en_US, "{:#Lo}", 00);
  test("01", en_US, "{:#Lo}", 01);
  test("01,000,000,000", en_US, "{:#Lo}", 01'000'000'000);

  test("-1", en_US, "{:Lo}", -01);
  test("-1,000,000,000", en_US, "{:Lo}", -01'000'000'000);

  test("-01", en_US, "{:#Lo}", -01);
  test("-01,000,000,000", en_US, "{:#Lo}", -01'000'000'000);

  std::locale::global(en_US);
  test("0", loc, "{:Lo}", 00);
  test("1", loc, "{:Lo}", 01);
  test("1_0_00_000_00_0", loc, "{:Lo}", 01'000'000'000);

  test("0", loc, "{:#Lo}", 00);
  test("01", loc, "{:#Lo}", 01);
  test("01_0_00_000_00_0", loc, "{:#Lo}", 01'000'000'000);

  test("-1", loc, "{:Lo}", -01);
  test("-1_0_00_000_00_0", loc, "{:Lo}", -01'000'000'000);

  test("-01", loc, "{:#Lo}", -01);
  test("-01_0_00_000_00_0", loc, "{:#Lo}", -01'000'000'000);

  // *** Hexadecimal ***
  std::locale::global(en_US);
  test("0", "{:Lx}", 0x0);
  test("1", "{:Lx}", 0x1);
  test("1,000,000,000", "{:Lx}", 0x1'000'000'000);

  test("0x0", "{:#Lx}", 0x0);
  test("0x1", "{:#Lx}", 0x1);
  test("0x1,000,000,000", "{:#Lx}", 0x1'000'000'000);

  test("-1", "{:LX}", -0x1);
  test("-1,000,000,000", "{:LX}", -0x1'000'000'000);

  test("-0X1", "{:#LX}", -0x1);
  test("-0X1,000,000,000", "{:#LX}", -0x1'000'000'000);

  std::locale::global(loc);
  test("0", "{:Lx}", 0x0);
  test("1", "{:Lx}", 0x1);
  test("1_0_00_000_00_0", "{:Lx}", 0x1'000'000'000);

  test("0x0", "{:#Lx}", 0x0);
  test("0x1", "{:#Lx}", 0x1);
  test("0x1_0_00_000_00_0", "{:#Lx}", 0x1'000'000'000);

  test("-1", "{:LX}", -0x1);
  test("-1_0_00_000_00_0", "{:LX}", -0x1'000'000'000);

  test("-0X1", "{:#LX}", -0x1);
  test("-0X1_0_00_000_00_0", "{:#LX}", -0x1'000'000'000);

  test("0", en_US, "{:Lx}", 0x0);
  test("1", en_US, "{:Lx}", 0x1);
  test("1,000,000,000", en_US, "{:Lx}", 0x1'000'000'000);

  test("0x0", en_US, "{:#Lx}", 0x0);
  test("0x1", en_US, "{:#Lx}", 0x1);
  test("0x1,000,000,000", en_US, "{:#Lx}", 0x1'000'000'000);

  test("-1", en_US, "{:LX}", -0x1);
  test("-1,000,000,000", en_US, "{:LX}", -0x1'000'000'000);

  test("-0X1", en_US, "{:#LX}", -0x1);
  test("-0X1,000,000,000", en_US, "{:#LX}", -0x1'000'000'000);

  std::locale::global(en_US);
  test("0", loc, "{:Lx}", 0x0);
  test("1", loc, "{:Lx}", 0x1);
  test("1_0_00_000_00_0", loc, "{:Lx}", 0x1'000'000'000);

  test("0x0", loc, "{:#Lx}", 0x0);
  test("0x1", loc, "{:#Lx}", 0x1);
  test("0x1_0_00_000_00_0", loc, "{:#Lx}", 0x1'000'000'000);

  test("-1", loc, "{:LX}", -0x1);
  test("-1_0_00_000_00_0", loc, "{:LX}", -0x1'000'000'000);

  test("-0X1", loc, "{:#LX}", -0x1);
  test("-0X1_0_00_000_00_0", loc, "{:#LX}", -0x1'000'000'000);

  // *** align-fill & width ***
  test("4_2", loc, "{:L}", 42);

  test("   4_2", loc, "{:6L}", 42);
  test("4_2   ", loc, "{:<6L}", 42);
  test(" 4_2  ", loc, "{:^6L}", 42);
  test("   4_2", loc, "{:>6L}", 42);

  test("4_2***", loc, "{:*<6L}", 42);
  test("*4_2**", loc, "{:*^6L}", 42);
  test("***4_2", loc, "{:*>6L}", 42);

  test("4_a*****", loc, "{:*<8Lx}", 0x4a);
  test("**4_a***", loc, "{:*^8Lx}", 0x4a);
  test("*****4_a", loc, "{:*>8Lx}", 0x4a);

  test("0x4_a***", loc, "{:*<#8Lx}", 0x4a);
  test("*0x4_a**", loc, "{:*^#8Lx}", 0x4a);
  test("***0x4_a", loc, "{:*>#8Lx}", 0x4a);

  test("4_A*****", loc, "{:*<8LX}", 0x4a);
  test("**4_A***", loc, "{:*^8LX}", 0x4a);
  test("*****4_A", loc, "{:*>8LX}", 0x4a);

  test("0X4_A***", loc, "{:*<#8LX}", 0x4a);
  test("*0X4_A**", loc, "{:*^#8LX}", 0x4a);
  test("***0X4_A", loc, "{:*>#8LX}", 0x4a);

  // Test whether zero padding is ignored
  test("4_2   ", loc, "{:<06L}", 42);
  test(" 4_2  ", loc, "{:^06L}", 42);
  test("   4_2", loc, "{:>06L}", 42);

  // *** zero-padding & width ***
  test("   4_2", loc, "{:6L}", 42);
  test("0004_2", loc, "{:06L}", 42);
  test("-004_2", loc, "{:06L}", -42);

  test("000004_a", loc, "{:08Lx}", 0x4a);
  test("0x0004_a", loc, "{:#08Lx}", 0x4a);
  test("0X0004_A", loc, "{:#08LX}", 0x4a);

  test("-00004_a", loc, "{:08Lx}", -0x4a);
  test("-0x004_a", loc, "{:#08Lx}", -0x4a);
  test("-0X004_A", loc, "{:#08LX}", -0x4a);
}

template <class F>
static void test_floating_point_hex_lower_case() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test("1.23456p-3", "{:La}", F(0x1.23456p-3));
  test("1.23456p-2", "{:La}", F(0x1.23456p-2));
  test("1.23456p-1", "{:La}", F(0x1.23456p-1));
  test("1.23456p+0", "{:La}", F(0x1.23456p0));
  test("1.23456p+1", "{:La}", F(0x1.23456p+1));
  test("1.23456p+2", "{:La}", F(0x1.23456p+2));
  test("1.23456p+3", "{:La}", F(0x1.23456p+3));
  test("1.23456p+20", "{:La}", F(0x1.23456p+20));

  std::locale::global(loc);
  test("1#23456p-3", "{:La}", F(0x1.23456p-3));
  test("1#23456p-2", "{:La}", F(0x1.23456p-2));
  test("1#23456p-1", "{:La}", F(0x1.23456p-1));
  test("1#23456p+0", "{:La}", F(0x1.23456p0));
  test("1#23456p+1", "{:La}", F(0x1.23456p+1));
  test("1#23456p+2", "{:La}", F(0x1.23456p+2));
  test("1#23456p+3", "{:La}", F(0x1.23456p+3));
  test("1#23456p+20", "{:La}", F(0x1.23456p+20));

  test("1.23456p-3", en_US, "{:La}", F(0x1.23456p-3));
  test("1.23456p-2", en_US, "{:La}", F(0x1.23456p-2));
  test("1.23456p-1", en_US, "{:La}", F(0x1.23456p-1));
  test("1.23456p+0", en_US, "{:La}", F(0x1.23456p0));
  test("1.23456p+1", en_US, "{:La}", F(0x1.23456p+1));
  test("1.23456p+2", en_US, "{:La}", F(0x1.23456p+2));
  test("1.23456p+3", en_US, "{:La}", F(0x1.23456p+3));
  test("1.23456p+20", en_US, "{:La}", F(0x1.23456p+20));

  std::locale::global(en_US);
  test("1#23456p-3", loc, "{:La}", F(0x1.23456p-3));
  test("1#23456p-2", loc, "{:La}", F(0x1.23456p-2));
  test("1#23456p-1", loc, "{:La}", F(0x1.23456p-1));
  test("1#23456p+0", loc, "{:La}", F(0x1.23456p0));
  test("1#23456p+1", loc, "{:La}", F(0x1.23456p+1));
  test("1#23456p+2", loc, "{:La}", F(0x1.23456p+2));
  test("1#23456p+3", loc, "{:La}", F(0x1.23456p+3));
  test("1#23456p+20", loc, "{:La}", F(0x1.23456p+20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test("1.23456p+3$$$", "{:$<13La}", F(0x1.23456p3));
  test("$$$1.23456p+3", "{:$>13La}", F(0x1.23456p3));
  test("$1.23456p+3$$", "{:$^13La}", F(0x1.23456p3));
  test("0001.23456p+3", "{:013La}", F(0x1.23456p3));
  test("-1.23456p+3$$$", "{:$<14La}", F(-0x1.23456p3));
  test("$$$-1.23456p+3", "{:$>14La}", F(-0x1.23456p3));
  test("$-1.23456p+3$$", "{:$^14La}", F(-0x1.23456p3));
  test("-0001.23456p+3", "{:014La}", F(-0x1.23456p3));

  std::locale::global(loc);
  test("1#23456p+3$$$", "{:$<13La}", F(0x1.23456p3));
  test("$$$1#23456p+3", "{:$>13La}", F(0x1.23456p3));
  test("$1#23456p+3$$", "{:$^13La}", F(0x1.23456p3));
  test("0001#23456p+3", "{:013La}", F(0x1.23456p3));
  test("-1#23456p+3$$$", "{:$<14La}", F(-0x1.23456p3));
  test("$$$-1#23456p+3", "{:$>14La}", F(-0x1.23456p3));
  test("$-1#23456p+3$$", "{:$^14La}", F(-0x1.23456p3));
  test("-0001#23456p+3", "{:014La}", F(-0x1.23456p3));

  test("1.23456p+3$$$", en_US, "{:$<13La}", F(0x1.23456p3));
  test("$$$1.23456p+3", en_US, "{:$>13La}", F(0x1.23456p3));
  test("$1.23456p+3$$", en_US, "{:$^13La}", F(0x1.23456p3));
  test("0001.23456p+3", en_US, "{:013La}", F(0x1.23456p3));
  test("-1.23456p+3$$$", en_US, "{:$<14La}", F(-0x1.23456p3));
  test("$$$-1.23456p+3", en_US, "{:$>14La}", F(-0x1.23456p3));
  test("$-1.23456p+3$$", en_US, "{:$^14La}", F(-0x1.23456p3));
  test("-0001.23456p+3", en_US, "{:014La}", F(-0x1.23456p3));

  std::locale::global(en_US);
  test("1#23456p+3$$$", loc, "{:$<13La}", F(0x1.23456p3));
  test("$$$1#23456p+3", loc, "{:$>13La}", F(0x1.23456p3));
  test("$1#23456p+3$$", loc, "{:$^13La}", F(0x1.23456p3));
  test("0001#23456p+3", loc, "{:013La}", F(0x1.23456p3));
  test("-1#23456p+3$$$", loc, "{:$<14La}", F(-0x1.23456p3));
  test("$$$-1#23456p+3", loc, "{:$>14La}", F(-0x1.23456p3));
  test("$-1#23456p+3$$", loc, "{:$^14La}", F(-0x1.23456p3));
  test("-0001#23456p+3", loc, "{:014La}", F(-0x1.23456p3));
}

template <class F>
static void test_floating_point_hex_upper_case() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test("1.23456P-3", "{:LA}", F(0x1.23456p-3));
  test("1.23456P-2", "{:LA}", F(0x1.23456p-2));
  test("1.23456P-1", "{:LA}", F(0x1.23456p-1));
  test("1.23456P+0", "{:LA}", F(0x1.23456p0));
  test("1.23456P+1", "{:LA}", F(0x1.23456p+1));
  test("1.23456P+2", "{:LA}", F(0x1.23456p+2));
  test("1.23456P+3", "{:LA}", F(0x1.23456p+3));
  test("1.23456P+20", "{:LA}", F(0x1.23456p+20));

  std::locale::global(loc);
  test("1#23456P-3", "{:LA}", F(0x1.23456p-3));
  test("1#23456P-2", "{:LA}", F(0x1.23456p-2));
  test("1#23456P-1", "{:LA}", F(0x1.23456p-1));
  test("1#23456P+0", "{:LA}", F(0x1.23456p0));
  test("1#23456P+1", "{:LA}", F(0x1.23456p+1));
  test("1#23456P+2", "{:LA}", F(0x1.23456p+2));
  test("1#23456P+3", "{:LA}", F(0x1.23456p+3));
  test("1#23456P+20", "{:LA}", F(0x1.23456p+20));

  test("1.23456P-3", en_US, "{:LA}", F(0x1.23456p-3));
  test("1.23456P-2", en_US, "{:LA}", F(0x1.23456p-2));
  test("1.23456P-1", en_US, "{:LA}", F(0x1.23456p-1));
  test("1.23456P+0", en_US, "{:LA}", F(0x1.23456p0));
  test("1.23456P+1", en_US, "{:LA}", F(0x1.23456p+1));
  test("1.23456P+2", en_US, "{:LA}", F(0x1.23456p+2));
  test("1.23456P+3", en_US, "{:LA}", F(0x1.23456p+3));
  test("1.23456P+20", en_US, "{:LA}", F(0x1.23456p+20));

  std::locale::global(en_US);
  test("1#23456P-3", loc, "{:LA}", F(0x1.23456p-3));
  test("1#23456P-2", loc, "{:LA}", F(0x1.23456p-2));
  test("1#23456P-1", loc, "{:LA}", F(0x1.23456p-1));
  test("1#23456P+0", loc, "{:LA}", F(0x1.23456p0));
  test("1#23456P+1", loc, "{:LA}", F(0x1.23456p+1));
  test("1#23456P+2", loc, "{:LA}", F(0x1.23456p+2));
  test("1#23456P+3", loc, "{:LA}", F(0x1.23456p+3));
  test("1#23456P+20", loc, "{:LA}", F(0x1.23456p+20));

  // *** Fill, align, zero Padding ***
  std::locale::global(en_US);
  test("1.23456P+3$$$", "{:$<13LA}", F(0x1.23456p3));
  test("$$$1.23456P+3", "{:$>13LA}", F(0x1.23456p3));
  test("$1.23456P+3$$", "{:$^13LA}", F(0x1.23456p3));
  test("0001.23456P+3", "{:013LA}", F(0x1.23456p3));
  test("-1.23456P+3$$$", "{:$<14LA}", F(-0x1.23456p3));
  test("$$$-1.23456P+3", "{:$>14LA}", F(-0x1.23456p3));
  test("$-1.23456P+3$$", "{:$^14LA}", F(-0x1.23456p3));
  test("-0001.23456P+3", "{:014LA}", F(-0x1.23456p3));

  std::locale::global(loc);
  test("1#23456P+3$$$", "{:$<13LA}", F(0x1.23456p3));
  test("$$$1#23456P+3", "{:$>13LA}", F(0x1.23456p3));
  test("$1#23456P+3$$", "{:$^13LA}", F(0x1.23456p3));
  test("0001#23456P+3", "{:013LA}", F(0x1.23456p3));
  test("-1#23456P+3$$$", "{:$<14LA}", F(-0x1.23456p3));
  test("$$$-1#23456P+3", "{:$>14LA}", F(-0x1.23456p3));
  test("$-1#23456P+3$$", "{:$^14LA}", F(-0x1.23456p3));
  test("-0001#23456P+3", "{:014LA}", F(-0x1.23456p3));

  test("1.23456P+3$$$", en_US, "{:$<13LA}", F(0x1.23456p3));
  test("$$$1.23456P+3", en_US, "{:$>13LA}", F(0x1.23456p3));
  test("$1.23456P+3$$", en_US, "{:$^13LA}", F(0x1.23456p3));
  test("0001.23456P+3", en_US, "{:013LA}", F(0x1.23456p3));
  test("-1.23456P+3$$$", en_US, "{:$<14LA}", F(-0x1.23456p3));
  test("$$$-1.23456P+3", en_US, "{:$>14LA}", F(-0x1.23456p3));
  test("$-1.23456P+3$$", en_US, "{:$^14LA}", F(-0x1.23456p3));
  test("-0001.23456P+3", en_US, "{:014LA}", F(-0x1.23456p3));

  std::locale::global(en_US);
  test("1#23456P+3$$$", loc, "{:$<13LA}", F(0x1.23456p3));
  test("$$$1#23456P+3", loc, "{:$>13LA}", F(0x1.23456p3));
  test("$1#23456P+3$$", loc, "{:$^13LA}", F(0x1.23456p3));
  test("0001#23456P+3", loc, "{:013LA}", F(0x1.23456p3));
  test("-1#23456P+3$$$", loc, "{:$<14LA}", F(-0x1.23456p3));
  test("$$$-1#23456P+3", loc, "{:$>14LA}", F(-0x1.23456p3));
  test("$-1#23456P+3$$", loc, "{:$^14LA}", F(-0x1.23456p3));
  test("-0001#23456P+3", loc, "{:014LA}", F(-0x1.23456p3));
}

template <class F>
static void test_floating_point_hex_lower_case_precision() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test("1.234560p-3", "{:.6La}", F(0x1.23456p-3));
  test("1.234560p-2", "{:.6La}", F(0x1.23456p-2));
  test("1.234560p-1", "{:.6La}", F(0x1.23456p-1));
  test("1.234560p+0", "{:.6La}", F(0x1.23456p0));
  test("1.234560p+1", "{:.6La}", F(0x1.23456p+1));
  test("1.234560p+2", "{:.6La}", F(0x1.23456p+2));
  test("1.234560p+3", "{:.6La}", F(0x1.23456p+3));
  test("1.234560p+20", "{:.6La}", F(0x1.23456p+20));

  std::locale::global(loc);
  test("1#234560p-3", "{:.6La}", F(0x1.23456p-3));
  test("1#234560p-2", "{:.6La}", F(0x1.23456p-2));
  test("1#234560p-1", "{:.6La}", F(0x1.23456p-1));
  test("1#234560p+0", "{:.6La}", F(0x1.23456p0));
  test("1#234560p+1", "{:.6La}", F(0x1.23456p+1));
  test("1#234560p+2", "{:.6La}", F(0x1.23456p+2));
  test("1#234560p+3", "{:.6La}", F(0x1.23456p+3));
  test("1#234560p+20", "{:.6La}", F(0x1.23456p+20));

  test("1.234560p-3", en_US, "{:.6La}", F(0x1.23456p-3));
  test("1.234560p-2", en_US, "{:.6La}", F(0x1.23456p-2));
  test("1.234560p-1", en_US, "{:.6La}", F(0x1.23456p-1));
  test("1.234560p+0", en_US, "{:.6La}", F(0x1.23456p0));
  test("1.234560p+1", en_US, "{:.6La}", F(0x1.23456p+1));
  test("1.234560p+2", en_US, "{:.6La}", F(0x1.23456p+2));
  test("1.234560p+3", en_US, "{:.6La}", F(0x1.23456p+3));
  test("1.234560p+20", en_US, "{:.6La}", F(0x1.23456p+20));

  std::locale::global(en_US);
  test("1#234560p-3", loc, "{:.6La}", F(0x1.23456p-3));
  test("1#234560p-2", loc, "{:.6La}", F(0x1.23456p-2));
  test("1#234560p-1", loc, "{:.6La}", F(0x1.23456p-1));
  test("1#234560p+0", loc, "{:.6La}", F(0x1.23456p0));
  test("1#234560p+1", loc, "{:.6La}", F(0x1.23456p+1));
  test("1#234560p+2", loc, "{:.6La}", F(0x1.23456p+2));
  test("1#234560p+3", loc, "{:.6La}", F(0x1.23456p+3));
  test("1#234560p+20", loc, "{:.6La}", F(0x1.23456p+20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test("1.234560p+3$$$", "{:$<14.6La}", F(0x1.23456p3));
  test("$$$1.234560p+3", "{:$>14.6La}", F(0x1.23456p3));
  test("$1.234560p+3$$", "{:$^14.6La}", F(0x1.23456p3));
  test("0001.234560p+3", "{:014.6La}", F(0x1.23456p3));
  test("-1.234560p+3$$$", "{:$<15.6La}", F(-0x1.23456p3));
  test("$$$-1.234560p+3", "{:$>15.6La}", F(-0x1.23456p3));
  test("$-1.234560p+3$$", "{:$^15.6La}", F(-0x1.23456p3));
  test("-0001.234560p+3", "{:015.6La}", F(-0x1.23456p3));

  std::locale::global(loc);
  test("1#234560p+3$$$", "{:$<14.6La}", F(0x1.23456p3));
  test("$$$1#234560p+3", "{:$>14.6La}", F(0x1.23456p3));
  test("$1#234560p+3$$", "{:$^14.6La}", F(0x1.23456p3));
  test("0001#234560p+3", "{:014.6La}", F(0x1.23456p3));
  test("-1#234560p+3$$$", "{:$<15.6La}", F(-0x1.23456p3));
  test("$$$-1#234560p+3", "{:$>15.6La}", F(-0x1.23456p3));
  test("$-1#234560p+3$$", "{:$^15.6La}", F(-0x1.23456p3));
  test("-0001#234560p+3", "{:015.6La}", F(-0x1.23456p3));

  test("1.234560p+3$$$", en_US, "{:$<14.6La}", F(0x1.23456p3));
  test("$$$1.234560p+3", en_US, "{:$>14.6La}", F(0x1.23456p3));
  test("$1.234560p+3$$", en_US, "{:$^14.6La}", F(0x1.23456p3));
  test("0001.234560p+3", en_US, "{:014.6La}", F(0x1.23456p3));
  test("-1.234560p+3$$$", en_US, "{:$<15.6La}", F(-0x1.23456p3));
  test("$$$-1.234560p+3", en_US, "{:$>15.6La}", F(-0x1.23456p3));
  test("$-1.234560p+3$$", en_US, "{:$^15.6La}", F(-0x1.23456p3));
  test("-0001.234560p+3", en_US, "{:015.6La}", F(-0x1.23456p3));

  std::locale::global(en_US);
  test("1#234560p+3$$$", loc, "{:$<14.6La}", F(0x1.23456p3));
  test("$$$1#234560p+3", loc, "{:$>14.6La}", F(0x1.23456p3));
  test("$1#234560p+3$$", loc, "{:$^14.6La}", F(0x1.23456p3));
  test("0001#234560p+3", loc, "{:014.6La}", F(0x1.23456p3));
  test("-1#234560p+3$$$", loc, "{:$<15.6La}", F(-0x1.23456p3));
  test("$$$-1#234560p+3", loc, "{:$>15.6La}", F(-0x1.23456p3));
  test("$-1#234560p+3$$", loc, "{:$^15.6La}", F(-0x1.23456p3));
  test("-0001#234560p+3", loc, "{:015.6La}", F(-0x1.23456p3));
}

template <class F>
static void test_floating_point_hex_upper_case_precision() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test("1.234560P-3", "{:.6LA}", F(0x1.23456p-3));
  test("1.234560P-2", "{:.6LA}", F(0x1.23456p-2));
  test("1.234560P-1", "{:.6LA}", F(0x1.23456p-1));
  test("1.234560P+0", "{:.6LA}", F(0x1.23456p0));
  test("1.234560P+1", "{:.6LA}", F(0x1.23456p+1));
  test("1.234560P+2", "{:.6LA}", F(0x1.23456p+2));
  test("1.234560P+3", "{:.6LA}", F(0x1.23456p+3));
  test("1.234560P+20", "{:.6LA}", F(0x1.23456p+20));

  std::locale::global(loc);
  test("1#234560P-3", "{:.6LA}", F(0x1.23456p-3));
  test("1#234560P-2", "{:.6LA}", F(0x1.23456p-2));
  test("1#234560P-1", "{:.6LA}", F(0x1.23456p-1));
  test("1#234560P+0", "{:.6LA}", F(0x1.23456p0));
  test("1#234560P+1", "{:.6LA}", F(0x1.23456p+1));
  test("1#234560P+2", "{:.6LA}", F(0x1.23456p+2));
  test("1#234560P+3", "{:.6LA}", F(0x1.23456p+3));
  test("1#234560P+20", "{:.6LA}", F(0x1.23456p+20));

  test("1.234560P-3", en_US, "{:.6LA}", F(0x1.23456p-3));
  test("1.234560P-2", en_US, "{:.6LA}", F(0x1.23456p-2));
  test("1.234560P-1", en_US, "{:.6LA}", F(0x1.23456p-1));
  test("1.234560P+0", en_US, "{:.6LA}", F(0x1.23456p0));
  test("1.234560P+1", en_US, "{:.6LA}", F(0x1.23456p+1));
  test("1.234560P+2", en_US, "{:.6LA}", F(0x1.23456p+2));
  test("1.234560P+3", en_US, "{:.6LA}", F(0x1.23456p+3));
  test("1.234560P+20", en_US, "{:.6LA}", F(0x1.23456p+20));

  std::locale::global(en_US);
  test("1#234560P-3", loc, "{:.6LA}", F(0x1.23456p-3));
  test("1#234560P-2", loc, "{:.6LA}", F(0x1.23456p-2));
  test("1#234560P-1", loc, "{:.6LA}", F(0x1.23456p-1));
  test("1#234560P+0", loc, "{:.6LA}", F(0x1.23456p0));
  test("1#234560P+1", loc, "{:.6LA}", F(0x1.23456p+1));
  test("1#234560P+2", loc, "{:.6LA}", F(0x1.23456p+2));
  test("1#234560P+3", loc, "{:.6LA}", F(0x1.23456p+3));
  test("1#234560P+20", loc, "{:.6LA}", F(0x1.23456p+20));

  // *** Fill, align, zero Padding ***
  std::locale::global(en_US);
  test("1.234560P+3$$$", "{:$<14.6LA}", F(0x1.23456p3));
  test("$$$1.234560P+3", "{:$>14.6LA}", F(0x1.23456p3));
  test("$1.234560P+3$$", "{:$^14.6LA}", F(0x1.23456p3));
  test("0001.234560P+3", "{:014.6LA}", F(0x1.23456p3));
  test("-1.234560P+3$$$", "{:$<15.6LA}", F(-0x1.23456p3));
  test("$$$-1.234560P+3", "{:$>15.6LA}", F(-0x1.23456p3));
  test("$-1.234560P+3$$", "{:$^15.6LA}", F(-0x1.23456p3));
  test("-0001.234560P+3", "{:015.6LA}", F(-0x1.23456p3));

  std::locale::global(loc);
  test("1#234560P+3$$$", "{:$<14.6LA}", F(0x1.23456p3));
  test("$$$1#234560P+3", "{:$>14.6LA}", F(0x1.23456p3));
  test("$1#234560P+3$$", "{:$^14.6LA}", F(0x1.23456p3));
  test("0001#234560P+3", "{:014.6LA}", F(0x1.23456p3));
  test("-1#234560P+3$$$", "{:$<15.6LA}", F(-0x1.23456p3));
  test("$$$-1#234560P+3", "{:$>15.6LA}", F(-0x1.23456p3));
  test("$-1#234560P+3$$", "{:$^15.6LA}", F(-0x1.23456p3));
  test("-0001#234560P+3", "{:015.6LA}", F(-0x1.23456p3));

  test("1.234560P+3$$$", en_US, "{:$<14.6LA}", F(0x1.23456p3));
  test("$$$1.234560P+3", en_US, "{:$>14.6LA}", F(0x1.23456p3));
  test("$1.234560P+3$$", en_US, "{:$^14.6LA}", F(0x1.23456p3));
  test("0001.234560P+3", en_US, "{:014.6LA}", F(0x1.23456p3));
  test("-1.234560P+3$$$", en_US, "{:$<15.6LA}", F(-0x1.23456p3));
  test("$$$-1.234560P+3", en_US, "{:$>15.6LA}", F(-0x1.23456p3));
  test("$-1.234560P+3$$", en_US, "{:$^15.6LA}", F(-0x1.23456p3));
  test("-0001.234560P+3", en_US, "{:015.6LA}", F(-0x1.23456p3));

  std::locale::global(en_US);
  test("1#234560P+3$$$", loc, "{:$<14.6LA}", F(0x1.23456p3));
  test("$$$1#234560P+3", loc, "{:$>14.6LA}", F(0x1.23456p3));
  test("$1#234560P+3$$", loc, "{:$^14.6LA}", F(0x1.23456p3));
  test("0001#234560P+3", loc, "{:014.6LA}", F(0x1.23456p3));
  test("-1#234560P+3$$$", loc, "{:$<15.6LA}", F(-0x1.23456p3));
  test("$$$-1#234560P+3", loc, "{:$>15.6LA}", F(-0x1.23456p3));
  test("$-1#234560P+3$$", loc, "{:$^15.6LA}", F(-0x1.23456p3));
  test("-0001#234560P+3", loc, "{:015.6LA}", F(-0x1.23456p3));
}

template <class F>
static void test_floating_point_scientific_lower_case() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test("1.234567e-03", "{:.6Le}", F(1.234567e-3));
  test("1.234567e-02", "{:.6Le}", F(1.234567e-2));
  test("1.234567e-01", "{:.6Le}", F(1.234567e-1));
  test("1.234567e+00", "{:.6Le}", F(1.234567e0));
  test("1.234567e+01", "{:.6Le}", F(1.234567e1));
  test("1.234567e+02", "{:.6Le}", F(1.234567e2));
  test("1.234567e+03", "{:.6Le}", F(1.234567e3));
  test("1.234567e+20", "{:.6Le}", F(1.234567e20));
  test("-1.234567e-03", "{:.6Le}", F(-1.234567e-3));
  test("-1.234567e-02", "{:.6Le}", F(-1.234567e-2));
  test("-1.234567e-01", "{:.6Le}", F(-1.234567e-1));
  test("-1.234567e+00", "{:.6Le}", F(-1.234567e0));
  test("-1.234567e+01", "{:.6Le}", F(-1.234567e1));
  test("-1.234567e+02", "{:.6Le}", F(-1.234567e2));
  test("-1.234567e+03", "{:.6Le}", F(-1.234567e3));
  test("-1.234567e+20", "{:.6Le}", F(-1.234567e20));

  std::locale::global(loc);
  test("1#234567e-03", "{:.6Le}", F(1.234567e-3));
  test("1#234567e-02", "{:.6Le}", F(1.234567e-2));
  test("1#234567e-01", "{:.6Le}", F(1.234567e-1));
  test("1#234567e+00", "{:.6Le}", F(1.234567e0));
  test("1#234567e+01", "{:.6Le}", F(1.234567e1));
  test("1#234567e+02", "{:.6Le}", F(1.234567e2));
  test("1#234567e+03", "{:.6Le}", F(1.234567e3));
  test("1#234567e+20", "{:.6Le}", F(1.234567e20));
  test("-1#234567e-03", "{:.6Le}", F(-1.234567e-3));
  test("-1#234567e-02", "{:.6Le}", F(-1.234567e-2));
  test("-1#234567e-01", "{:.6Le}", F(-1.234567e-1));
  test("-1#234567e+00", "{:.6Le}", F(-1.234567e0));
  test("-1#234567e+01", "{:.6Le}", F(-1.234567e1));
  test("-1#234567e+02", "{:.6Le}", F(-1.234567e2));
  test("-1#234567e+03", "{:.6Le}", F(-1.234567e3));
  test("-1#234567e+20", "{:.6Le}", F(-1.234567e20));

  test("1.234567e-03", en_US, "{:.6Le}", F(1.234567e-3));
  test("1.234567e-02", en_US, "{:.6Le}", F(1.234567e-2));
  test("1.234567e-01", en_US, "{:.6Le}", F(1.234567e-1));
  test("1.234567e+00", en_US, "{:.6Le}", F(1.234567e0));
  test("1.234567e+01", en_US, "{:.6Le}", F(1.234567e1));
  test("1.234567e+02", en_US, "{:.6Le}", F(1.234567e2));
  test("1.234567e+03", en_US, "{:.6Le}", F(1.234567e3));
  test("1.234567e+20", en_US, "{:.6Le}", F(1.234567e20));
  test("-1.234567e-03", en_US, "{:.6Le}", F(-1.234567e-3));
  test("-1.234567e-02", en_US, "{:.6Le}", F(-1.234567e-2));
  test("-1.234567e-01", en_US, "{:.6Le}", F(-1.234567e-1));
  test("-1.234567e+00", en_US, "{:.6Le}", F(-1.234567e0));
  test("-1.234567e+01", en_US, "{:.6Le}", F(-1.234567e1));
  test("-1.234567e+02", en_US, "{:.6Le}", F(-1.234567e2));
  test("-1.234567e+03", en_US, "{:.6Le}", F(-1.234567e3));
  test("-1.234567e+20", en_US, "{:.6Le}", F(-1.234567e20));

  std::locale::global(en_US);
  test("1#234567e-03", loc, "{:.6Le}", F(1.234567e-3));
  test("1#234567e-02", loc, "{:.6Le}", F(1.234567e-2));
  test("1#234567e-01", loc, "{:.6Le}", F(1.234567e-1));
  test("1#234567e+00", loc, "{:.6Le}", F(1.234567e0));
  test("1#234567e+01", loc, "{:.6Le}", F(1.234567e1));
  test("1#234567e+02", loc, "{:.6Le}", F(1.234567e2));
  test("1#234567e+03", loc, "{:.6Le}", F(1.234567e3));
  test("1#234567e+20", loc, "{:.6Le}", F(1.234567e20));
  test("-1#234567e-03", loc, "{:.6Le}", F(-1.234567e-3));
  test("-1#234567e-02", loc, "{:.6Le}", F(-1.234567e-2));
  test("-1#234567e-01", loc, "{:.6Le}", F(-1.234567e-1));
  test("-1#234567e+00", loc, "{:.6Le}", F(-1.234567e0));
  test("-1#234567e+01", loc, "{:.6Le}", F(-1.234567e1));
  test("-1#234567e+02", loc, "{:.6Le}", F(-1.234567e2));
  test("-1#234567e+03", loc, "{:.6Le}", F(-1.234567e3));
  test("-1#234567e+20", loc, "{:.6Le}", F(-1.234567e20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test("1.234567e+03$$$", "{:$<15.6Le}", F(1.234567e3));
  test("$$$1.234567e+03", "{:$>15.6Le}", F(1.234567e3));
  test("$1.234567e+03$$", "{:$^15.6Le}", F(1.234567e3));
  test("0001.234567e+03", "{:015.6Le}", F(1.234567e3));
  test("-1.234567e+03$$$", "{:$<16.6Le}", F(-1.234567e3));
  test("$$$-1.234567e+03", "{:$>16.6Le}", F(-1.234567e3));
  test("$-1.234567e+03$$", "{:$^16.6Le}", F(-1.234567e3));
  test("-0001.234567e+03", "{:016.6Le}", F(-1.234567e3));

  std::locale::global(loc);
  test("1#234567e+03$$$", "{:$<15.6Le}", F(1.234567e3));
  test("$$$1#234567e+03", "{:$>15.6Le}", F(1.234567e3));
  test("$1#234567e+03$$", "{:$^15.6Le}", F(1.234567e3));
  test("0001#234567e+03", "{:015.6Le}", F(1.234567e3));
  test("-1#234567e+03$$$", "{:$<16.6Le}", F(-1.234567e3));
  test("$$$-1#234567e+03", "{:$>16.6Le}", F(-1.234567e3));
  test("$-1#234567e+03$$", "{:$^16.6Le}", F(-1.234567e3));
  test("-0001#234567e+03", "{:016.6Le}", F(-1.234567e3));

  test("1.234567e+03$$$", en_US, "{:$<15.6Le}", F(1.234567e3));
  test("$$$1.234567e+03", en_US, "{:$>15.6Le}", F(1.234567e3));
  test("$1.234567e+03$$", en_US, "{:$^15.6Le}", F(1.234567e3));
  test("0001.234567e+03", en_US, "{:015.6Le}", F(1.234567e3));
  test("-1.234567e+03$$$", en_US, "{:$<16.6Le}", F(-1.234567e3));
  test("$$$-1.234567e+03", en_US, "{:$>16.6Le}", F(-1.234567e3));
  test("$-1.234567e+03$$", en_US, "{:$^16.6Le}", F(-1.234567e3));
  test("-0001.234567e+03", en_US, "{:016.6Le}", F(-1.234567e3));

  std::locale::global(en_US);
  test("1#234567e+03$$$", loc, "{:$<15.6Le}", F(1.234567e3));
  test("$$$1#234567e+03", loc, "{:$>15.6Le}", F(1.234567e3));
  test("$1#234567e+03$$", loc, "{:$^15.6Le}", F(1.234567e3));
  test("0001#234567e+03", loc, "{:015.6Le}", F(1.234567e3));
  test("-1#234567e+03$$$", loc, "{:$<16.6Le}", F(-1.234567e3));
  test("$$$-1#234567e+03", loc, "{:$>16.6Le}", F(-1.234567e3));
  test("$-1#234567e+03$$", loc, "{:$^16.6Le}", F(-1.234567e3));
  test("-0001#234567e+03", loc, "{:016.6Le}", F(-1.234567e3));
}

template <class F>
static void test_floating_point_scientific_upper_case() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test("1.234567E-03", "{:.6LE}", F(1.234567e-3));
  test("1.234567E-02", "{:.6LE}", F(1.234567e-2));
  test("1.234567E-01", "{:.6LE}", F(1.234567e-1));
  test("1.234567E+00", "{:.6LE}", F(1.234567e0));
  test("1.234567E+01", "{:.6LE}", F(1.234567e1));
  test("1.234567E+02", "{:.6LE}", F(1.234567e2));
  test("1.234567E+03", "{:.6LE}", F(1.234567e3));
  test("1.234567E+20", "{:.6LE}", F(1.234567e20));
  test("-1.234567E-03", "{:.6LE}", F(-1.234567e-3));
  test("-1.234567E-02", "{:.6LE}", F(-1.234567e-2));
  test("-1.234567E-01", "{:.6LE}", F(-1.234567e-1));
  test("-1.234567E+00", "{:.6LE}", F(-1.234567e0));
  test("-1.234567E+01", "{:.6LE}", F(-1.234567e1));
  test("-1.234567E+02", "{:.6LE}", F(-1.234567e2));
  test("-1.234567E+03", "{:.6LE}", F(-1.234567e3));
  test("-1.234567E+20", "{:.6LE}", F(-1.234567e20));

  std::locale::global(loc);
  test("1#234567E-03", "{:.6LE}", F(1.234567e-3));
  test("1#234567E-02", "{:.6LE}", F(1.234567e-2));
  test("1#234567E-01", "{:.6LE}", F(1.234567e-1));
  test("1#234567E+00", "{:.6LE}", F(1.234567e0));
  test("1#234567E+01", "{:.6LE}", F(1.234567e1));
  test("1#234567E+02", "{:.6LE}", F(1.234567e2));
  test("1#234567E+03", "{:.6LE}", F(1.234567e3));
  test("1#234567E+20", "{:.6LE}", F(1.234567e20));
  test("-1#234567E-03", "{:.6LE}", F(-1.234567e-3));
  test("-1#234567E-02", "{:.6LE}", F(-1.234567e-2));
  test("-1#234567E-01", "{:.6LE}", F(-1.234567e-1));
  test("-1#234567E+00", "{:.6LE}", F(-1.234567e0));
  test("-1#234567E+01", "{:.6LE}", F(-1.234567e1));
  test("-1#234567E+02", "{:.6LE}", F(-1.234567e2));
  test("-1#234567E+03", "{:.6LE}", F(-1.234567e3));
  test("-1#234567E+20", "{:.6LE}", F(-1.234567e20));

  test("1.234567E-03", en_US, "{:.6LE}", F(1.234567e-3));
  test("1.234567E-02", en_US, "{:.6LE}", F(1.234567e-2));
  test("1.234567E-01", en_US, "{:.6LE}", F(1.234567e-1));
  test("1.234567E+00", en_US, "{:.6LE}", F(1.234567e0));
  test("1.234567E+01", en_US, "{:.6LE}", F(1.234567e1));
  test("1.234567E+02", en_US, "{:.6LE}", F(1.234567e2));
  test("1.234567E+03", en_US, "{:.6LE}", F(1.234567e3));
  test("1.234567E+20", en_US, "{:.6LE}", F(1.234567e20));
  test("-1.234567E-03", en_US, "{:.6LE}", F(-1.234567e-3));
  test("-1.234567E-02", en_US, "{:.6LE}", F(-1.234567e-2));
  test("-1.234567E-01", en_US, "{:.6LE}", F(-1.234567e-1));
  test("-1.234567E+00", en_US, "{:.6LE}", F(-1.234567e0));
  test("-1.234567E+01", en_US, "{:.6LE}", F(-1.234567e1));
  test("-1.234567E+02", en_US, "{:.6LE}", F(-1.234567e2));
  test("-1.234567E+03", en_US, "{:.6LE}", F(-1.234567e3));
  test("-1.234567E+20", en_US, "{:.6LE}", F(-1.234567e20));

  std::locale::global(en_US);
  test("1#234567E-03", loc, "{:.6LE}", F(1.234567e-3));
  test("1#234567E-02", loc, "{:.6LE}", F(1.234567e-2));
  test("1#234567E-01", loc, "{:.6LE}", F(1.234567e-1));
  test("1#234567E+00", loc, "{:.6LE}", F(1.234567e0));
  test("1#234567E+01", loc, "{:.6LE}", F(1.234567e1));
  test("1#234567E+02", loc, "{:.6LE}", F(1.234567e2));
  test("1#234567E+03", loc, "{:.6LE}", F(1.234567e3));
  test("1#234567E+20", loc, "{:.6LE}", F(1.234567e20));
  test("-1#234567E-03", loc, "{:.6LE}", F(-1.234567e-3));
  test("-1#234567E-02", loc, "{:.6LE}", F(-1.234567e-2));
  test("-1#234567E-01", loc, "{:.6LE}", F(-1.234567e-1));
  test("-1#234567E+00", loc, "{:.6LE}", F(-1.234567e0));
  test("-1#234567E+01", loc, "{:.6LE}", F(-1.234567e1));
  test("-1#234567E+02", loc, "{:.6LE}", F(-1.234567e2));
  test("-1#234567E+03", loc, "{:.6LE}", F(-1.234567e3));
  test("-1#234567E+20", loc, "{:.6LE}", F(-1.234567e20));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test("1.234567E+03$$$", "{:$<15.6LE}", F(1.234567e3));
  test("$$$1.234567E+03", "{:$>15.6LE}", F(1.234567e3));
  test("$1.234567E+03$$", "{:$^15.6LE}", F(1.234567e3));
  test("0001.234567E+03", "{:015.6LE}", F(1.234567e3));
  test("-1.234567E+03$$$", "{:$<16.6LE}", F(-1.234567e3));
  test("$$$-1.234567E+03", "{:$>16.6LE}", F(-1.234567e3));
  test("$-1.234567E+03$$", "{:$^16.6LE}", F(-1.234567e3));
  test("-0001.234567E+03", "{:016.6LE}", F(-1.234567e3));

  std::locale::global(loc);
  test("1#234567E+03$$$", "{:$<15.6LE}", F(1.234567e3));
  test("$$$1#234567E+03", "{:$>15.6LE}", F(1.234567e3));
  test("$1#234567E+03$$", "{:$^15.6LE}", F(1.234567e3));
  test("0001#234567E+03", "{:015.6LE}", F(1.234567e3));
  test("-1#234567E+03$$$", "{:$<16.6LE}", F(-1.234567e3));
  test("$$$-1#234567E+03", "{:$>16.6LE}", F(-1.234567e3));
  test("$-1#234567E+03$$", "{:$^16.6LE}", F(-1.234567e3));
  test("-0001#234567E+03", "{:016.6LE}", F(-1.234567e3));

  test("1.234567E+03$$$", en_US, "{:$<15.6LE}", F(1.234567e3));
  test("$$$1.234567E+03", en_US, "{:$>15.6LE}", F(1.234567e3));
  test("$1.234567E+03$$", en_US, "{:$^15.6LE}", F(1.234567e3));
  test("0001.234567E+03", en_US, "{:015.6LE}", F(1.234567e3));
  test("-1.234567E+03$$$", en_US, "{:$<16.6LE}", F(-1.234567e3));
  test("$$$-1.234567E+03", en_US, "{:$>16.6LE}", F(-1.234567e3));
  test("$-1.234567E+03$$", en_US, "{:$^16.6LE}", F(-1.234567e3));
  test("-0001.234567E+03", en_US, "{:016.6LE}", F(-1.234567e3));

  std::locale::global(en_US);
  test("1#234567E+03$$$", loc, "{:$<15.6LE}", F(1.234567e3));
  test("$$$1#234567E+03", loc, "{:$>15.6LE}", F(1.234567e3));
  test("$1#234567E+03$$", loc, "{:$^15.6LE}", F(1.234567e3));
  test("0001#234567E+03", loc, "{:015.6LE}", F(1.234567e3));
  test("-1#234567E+03$$$", loc, "{:$<16.6LE}", F(-1.234567e3));
  test("$$$-1#234567E+03", loc, "{:$>16.6LE}", F(-1.234567e3));
  test("$-1#234567E+03$$", loc, "{:$^16.6LE}", F(-1.234567e3));
  test("-0001#234567E+03", loc, "{:016.6LE}", F(-1.234567e3));
}

template <class F>
static void test_floating_point_fixed_lower_case() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test("0.000001", "{:.6Lf}", F(1.234567e-6));
  test("0.000012", "{:.6Lf}", F(1.234567e-5));
  test("0.000123", "{:.6Lf}", F(1.234567e-4));
  test("0.001235", "{:.6Lf}", F(1.234567e-3));
  test("0.012346", "{:.6Lf}", F(1.234567e-2));
  test("0.123457", "{:.6Lf}", F(1.234567e-1));
  test("1.234567", "{:.6Lf}", F(1.234567e0));
  test("12.345670", "{:.6Lf}", F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("123.456700", "{:.6Lf}", F(1.234567e2));
    test("1,234.567000", "{:.6Lf}", F(1.234567e3));
    test("12,345.670000", "{:.6Lf}", F(1.234567e4));
    test("123,456.700000", "{:.6Lf}", F(1.234567e5));
    test("1,234,567.000000", "{:.6Lf}", F(1.234567e6));
    test("12,345,670.000000", "{:.6Lf}", F(1.234567e7));
    test("123,456,700,000,000,000,000.000000", "{:.6Lf}", F(1.234567e20));
  }
  test("-0.000001", "{:.6Lf}", F(-1.234567e-6));
  test("-0.000012", "{:.6Lf}", F(-1.234567e-5));
  test("-0.000123", "{:.6Lf}", F(-1.234567e-4));
  test("-0.001235", "{:.6Lf}", F(-1.234567e-3));
  test("-0.012346", "{:.6Lf}", F(-1.234567e-2));
  test("-0.123457", "{:.6Lf}", F(-1.234567e-1));
  test("-1.234567", "{:.6Lf}", F(-1.234567e0));
  test("-12.345670", "{:.6Lf}", F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("-123.456700", "{:.6Lf}", F(-1.234567e2));
    test("-1,234.567000", "{:.6Lf}", F(-1.234567e3));
    test("-12,345.670000", "{:.6Lf}", F(-1.234567e4));
    test("-123,456.700000", "{:.6Lf}", F(-1.234567e5));
    test("-1,234,567.000000", "{:.6Lf}", F(-1.234567e6));
    test("-12,345,670.000000", "{:.6Lf}", F(-1.234567e7));
    test("-123,456,700,000,000,000,000.000000", "{:.6Lf}", F(-1.234567e20));
  }

  std::locale::global(loc);
  test("0#000001", "{:.6Lf}", F(1.234567e-6));
  test("0#000012", "{:.6Lf}", F(1.234567e-5));
  test("0#000123", "{:.6Lf}", F(1.234567e-4));
  test("0#001235", "{:.6Lf}", F(1.234567e-3));
  test("0#012346", "{:.6Lf}", F(1.234567e-2));
  test("0#123457", "{:.6Lf}", F(1.234567e-1));
  test("1#234567", "{:.6Lf}", F(1.234567e0));
  test("1_2#345670", "{:.6Lf}", F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("12_3#456700", "{:.6Lf}", F(1.234567e2));
    test("1_23_4#567000", "{:.6Lf}", F(1.234567e3));
    test("12_34_5#670000", "{:.6Lf}", F(1.234567e4));
    test("123_45_6#700000", "{:.6Lf}", F(1.234567e5));
    test("1_234_56_7#000000", "{:.6Lf}", F(1.234567e6));
    test("12_345_67_0#000000", "{:.6Lf}", F(1.234567e7));
    test("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000", "{:.6Lf}", F(1.234567e20));
  }
  test("-0#000001", "{:.6Lf}", F(-1.234567e-6));
  test("-0#000012", "{:.6Lf}", F(-1.234567e-5));
  test("-0#000123", "{:.6Lf}", F(-1.234567e-4));
  test("-0#001235", "{:.6Lf}", F(-1.234567e-3));
  test("-0#012346", "{:.6Lf}", F(-1.234567e-2));
  test("-0#123457", "{:.6Lf}", F(-1.234567e-1));
  test("-1#234567", "{:.6Lf}", F(-1.234567e0));
  test("-1_2#345670", "{:.6Lf}", F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("-12_3#456700", "{:.6Lf}", F(-1.234567e2));
    test("-1_23_4#567000", "{:.6Lf}", F(-1.234567e3));
    test("-12_34_5#670000", "{:.6Lf}", F(-1.234567e4));
    test("-123_45_6#700000", "{:.6Lf}", F(-1.234567e5));
    test("-1_234_56_7#000000", "{:.6Lf}", F(-1.234567e6));
    test("-12_345_67_0#000000", "{:.6Lf}", F(-1.234567e7));
    test("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000", "{:.6Lf}", F(-1.234567e20));
  }

  test("0.000001", en_US, "{:.6Lf}", F(1.234567e-6));
  test("0.000012", en_US, "{:.6Lf}", F(1.234567e-5));
  test("0.000123", en_US, "{:.6Lf}", F(1.234567e-4));
  test("0.001235", en_US, "{:.6Lf}", F(1.234567e-3));
  test("0.012346", en_US, "{:.6Lf}", F(1.234567e-2));
  test("0.123457", en_US, "{:.6Lf}", F(1.234567e-1));
  test("1.234567", en_US, "{:.6Lf}", F(1.234567e0));
  test("12.345670", en_US, "{:.6Lf}", F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("123.456700", en_US, "{:.6Lf}", F(1.234567e2));
    test("1,234.567000", en_US, "{:.6Lf}", F(1.234567e3));
    test("12,345.670000", en_US, "{:.6Lf}", F(1.234567e4));
    test("123,456.700000", en_US, "{:.6Lf}", F(1.234567e5));
    test("1,234,567.000000", en_US, "{:.6Lf}", F(1.234567e6));
    test("12,345,670.000000", en_US, "{:.6Lf}", F(1.234567e7));
    test("123,456,700,000,000,000,000.000000", en_US, "{:.6Lf}", F(1.234567e20));
  }
  test("-0.000001", en_US, "{:.6Lf}", F(-1.234567e-6));
  test("-0.000012", en_US, "{:.6Lf}", F(-1.234567e-5));
  test("-0.000123", en_US, "{:.6Lf}", F(-1.234567e-4));
  test("-0.001235", en_US, "{:.6Lf}", F(-1.234567e-3));
  test("-0.012346", en_US, "{:.6Lf}", F(-1.234567e-2));
  test("-0.123457", en_US, "{:.6Lf}", F(-1.234567e-1));
  test("-1.234567", en_US, "{:.6Lf}", F(-1.234567e0));
  test("-12.345670", en_US, "{:.6Lf}", F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("-123.456700", en_US, "{:.6Lf}", F(-1.234567e2));
    test("-1,234.567000", en_US, "{:.6Lf}", F(-1.234567e3));
    test("-12,345.670000", en_US, "{:.6Lf}", F(-1.234567e4));
    test("-123,456.700000", en_US, "{:.6Lf}", F(-1.234567e5));
    test("-1,234,567.000000", en_US, "{:.6Lf}", F(-1.234567e6));
    test("-12,345,670.000000", en_US, "{:.6Lf}", F(-1.234567e7));
    test("-123,456,700,000,000,000,000.000000", en_US, "{:.6Lf}", F(-1.234567e20));
  }

  std::locale::global(en_US);
  test("0#000001", loc, "{:.6Lf}", F(1.234567e-6));
  test("0#000012", loc, "{:.6Lf}", F(1.234567e-5));
  test("0#000123", loc, "{:.6Lf}", F(1.234567e-4));
  test("0#001235", loc, "{:.6Lf}", F(1.234567e-3));
  test("0#012346", loc, "{:.6Lf}", F(1.234567e-2));
  test("0#123457", loc, "{:.6Lf}", F(1.234567e-1));
  test("1#234567", loc, "{:.6Lf}", F(1.234567e0));
  test("1_2#345670", loc, "{:.6Lf}", F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("12_3#456700", loc, "{:.6Lf}", F(1.234567e2));
    test("1_23_4#567000", loc, "{:.6Lf}", F(1.234567e3));
    test("12_34_5#670000", loc, "{:.6Lf}", F(1.234567e4));
    test("123_45_6#700000", loc, "{:.6Lf}", F(1.234567e5));
    test("1_234_56_7#000000", loc, "{:.6Lf}", F(1.234567e6));
    test("12_345_67_0#000000", loc, "{:.6Lf}", F(1.234567e7));
    test("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000", loc, "{:.6Lf}", F(1.234567e20));
  }
  test("-0#000001", loc, "{:.6Lf}", F(-1.234567e-6));
  test("-0#000012", loc, "{:.6Lf}", F(-1.234567e-5));
  test("-0#000123", loc, "{:.6Lf}", F(-1.234567e-4));
  test("-0#001235", loc, "{:.6Lf}", F(-1.234567e-3));
  test("-0#012346", loc, "{:.6Lf}", F(-1.234567e-2));
  test("-0#123457", loc, "{:.6Lf}", F(-1.234567e-1));
  test("-1#234567", loc, "{:.6Lf}", F(-1.234567e0));
  test("-1_2#345670", loc, "{:.6Lf}", F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("-12_3#456700", loc, "{:.6Lf}", F(-1.234567e2));
    test("-1_23_4#567000", loc, "{:.6Lf}", F(-1.234567e3));
    test("-12_34_5#670000", loc, "{:.6Lf}", F(-1.234567e4));
    test("-123_45_6#700000", loc, "{:.6Lf}", F(-1.234567e5));
    test("-1_234_56_7#000000", loc, "{:.6Lf}", F(-1.234567e6));
    test("-12_345_67_0#000000", loc, "{:.6Lf}", F(-1.234567e7));
    test("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000", loc, "{:.6Lf}", F(-1.234567e20));
  }

  // *** Fill, align, zero padding ***
  if constexpr (sizeof(F) > sizeof(float)) {
    std::locale::global(en_US);
    test("1,234.567000$$$", "{:$<15.6Lf}", F(1.234567e3));
    test("$$$1,234.567000", "{:$>15.6Lf}", F(1.234567e3));
    test("$1,234.567000$$", "{:$^15.6Lf}", F(1.234567e3));
    test("0001,234.567000", "{:015.6Lf}", F(1.234567e3));
    test("-1,234.567000$$$", "{:$<16.6Lf}", F(-1.234567e3));
    test("$$$-1,234.567000", "{:$>16.6Lf}", F(-1.234567e3));
    test("$-1,234.567000$$", "{:$^16.6Lf}", F(-1.234567e3));
    test("-0001,234.567000", "{:016.6Lf}", F(-1.234567e3));

    std::locale::global(loc);
    test("1_23_4#567000$$$", "{:$<16.6Lf}", F(1.234567e3));
    test("$$$1_23_4#567000", "{:$>16.6Lf}", F(1.234567e3));
    test("$1_23_4#567000$$", "{:$^16.6Lf}", F(1.234567e3));
    test("0001_23_4#567000", "{:016.6Lf}", F(1.234567e3));
    test("-1_23_4#567000$$$", "{:$<17.6Lf}", F(-1.234567e3));
    test("$$$-1_23_4#567000", "{:$>17.6Lf}", F(-1.234567e3));
    test("$-1_23_4#567000$$", "{:$^17.6Lf}", F(-1.234567e3));
    test("-0001_23_4#567000", "{:017.6Lf}", F(-1.234567e3));

    test("1,234.567000$$$", en_US, "{:$<15.6Lf}", F(1.234567e3));
    test("$$$1,234.567000", en_US, "{:$>15.6Lf}", F(1.234567e3));
    test("$1,234.567000$$", en_US, "{:$^15.6Lf}", F(1.234567e3));
    test("0001,234.567000", en_US, "{:015.6Lf}", F(1.234567e3));
    test("-1,234.567000$$$", en_US, "{:$<16.6Lf}", F(-1.234567e3));
    test("$$$-1,234.567000", en_US, "{:$>16.6Lf}", F(-1.234567e3));
    test("$-1,234.567000$$", en_US, "{:$^16.6Lf}", F(-1.234567e3));
    test("-0001,234.567000", en_US, "{:016.6Lf}", F(-1.234567e3));

    std::locale::global(en_US);
    test("1_23_4#567000$$$", loc, "{:$<16.6Lf}", F(1.234567e3));
    test("$$$1_23_4#567000", loc, "{:$>16.6Lf}", F(1.234567e3));
    test("$1_23_4#567000$$", loc, "{:$^16.6Lf}", F(1.234567e3));
    test("0001_23_4#567000", loc, "{:016.6Lf}", F(1.234567e3));
    test("-1_23_4#567000$$$", loc, "{:$<17.6Lf}", F(-1.234567e3));
    test("$$$-1_23_4#567000", loc, "{:$>17.6Lf}", F(-1.234567e3));
    test("$-1_23_4#567000$$", loc, "{:$^17.6Lf}", F(-1.234567e3));
    test("-0001_23_4#567000", loc, "{:017.6Lf}", F(-1.234567e3));
  }
}

template <class F>
static void test_floating_point_fixed_upper_case() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test("0.000001", "{:.6Lf}", F(1.234567e-6));
  test("0.000012", "{:.6Lf}", F(1.234567e-5));
  test("0.000123", "{:.6Lf}", F(1.234567e-4));
  test("0.001235", "{:.6Lf}", F(1.234567e-3));
  test("0.012346", "{:.6Lf}", F(1.234567e-2));
  test("0.123457", "{:.6Lf}", F(1.234567e-1));
  test("1.234567", "{:.6Lf}", F(1.234567e0));
  test("12.345670", "{:.6Lf}", F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("123.456700", "{:.6Lf}", F(1.234567e2));
    test("1,234.567000", "{:.6Lf}", F(1.234567e3));
    test("12,345.670000", "{:.6Lf}", F(1.234567e4));
    test("123,456.700000", "{:.6Lf}", F(1.234567e5));
    test("1,234,567.000000", "{:.6Lf}", F(1.234567e6));
    test("12,345,670.000000", "{:.6Lf}", F(1.234567e7));
    test("123,456,700,000,000,000,000.000000", "{:.6Lf}", F(1.234567e20));
  }
  test("-0.000001", "{:.6Lf}", F(-1.234567e-6));
  test("-0.000012", "{:.6Lf}", F(-1.234567e-5));
  test("-0.000123", "{:.6Lf}", F(-1.234567e-4));
  test("-0.001235", "{:.6Lf}", F(-1.234567e-3));
  test("-0.012346", "{:.6Lf}", F(-1.234567e-2));
  test("-0.123457", "{:.6Lf}", F(-1.234567e-1));
  test("-1.234567", "{:.6Lf}", F(-1.234567e0));
  test("-12.345670", "{:.6Lf}", F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("-123.456700", "{:.6Lf}", F(-1.234567e2));
    test("-1,234.567000", "{:.6Lf}", F(-1.234567e3));
    test("-12,345.670000", "{:.6Lf}", F(-1.234567e4));
    test("-123,456.700000", "{:.6Lf}", F(-1.234567e5));
    test("-1,234,567.000000", "{:.6Lf}", F(-1.234567e6));
    test("-12,345,670.000000", "{:.6Lf}", F(-1.234567e7));
    test("-123,456,700,000,000,000,000.000000", "{:.6Lf}", F(-1.234567e20));
  }

  std::locale::global(loc);
  test("0#000001", "{:.6Lf}", F(1.234567e-6));
  test("0#000012", "{:.6Lf}", F(1.234567e-5));
  test("0#000123", "{:.6Lf}", F(1.234567e-4));
  test("0#001235", "{:.6Lf}", F(1.234567e-3));
  test("0#012346", "{:.6Lf}", F(1.234567e-2));
  test("0#123457", "{:.6Lf}", F(1.234567e-1));
  test("1#234567", "{:.6Lf}", F(1.234567e0));
  test("1_2#345670", "{:.6Lf}", F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("12_3#456700", "{:.6Lf}", F(1.234567e2));
    test("1_23_4#567000", "{:.6Lf}", F(1.234567e3));
    test("12_34_5#670000", "{:.6Lf}", F(1.234567e4));
    test("123_45_6#700000", "{:.6Lf}", F(1.234567e5));
    test("1_234_56_7#000000", "{:.6Lf}", F(1.234567e6));
    test("12_345_67_0#000000", "{:.6Lf}", F(1.234567e7));
    test("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000", "{:.6Lf}", F(1.234567e20));
  }
  test("-0#000001", "{:.6Lf}", F(-1.234567e-6));
  test("-0#000012", "{:.6Lf}", F(-1.234567e-5));
  test("-0#000123", "{:.6Lf}", F(-1.234567e-4));
  test("-0#001235", "{:.6Lf}", F(-1.234567e-3));
  test("-0#012346", "{:.6Lf}", F(-1.234567e-2));
  test("-0#123457", "{:.6Lf}", F(-1.234567e-1));
  test("-1#234567", "{:.6Lf}", F(-1.234567e0));
  test("-1_2#345670", "{:.6Lf}", F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("-12_3#456700", "{:.6Lf}", F(-1.234567e2));
    test("-1_23_4#567000", "{:.6Lf}", F(-1.234567e3));
    test("-12_34_5#670000", "{:.6Lf}", F(-1.234567e4));
    test("-123_45_6#700000", "{:.6Lf}", F(-1.234567e5));
    test("-1_234_56_7#000000", "{:.6Lf}", F(-1.234567e6));
    test("-12_345_67_0#000000", "{:.6Lf}", F(-1.234567e7));
    test("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000", "{:.6Lf}", F(-1.234567e20));
  }

  test("0.000001", en_US, "{:.6Lf}", F(1.234567e-6));
  test("0.000012", en_US, "{:.6Lf}", F(1.234567e-5));
  test("0.000123", en_US, "{:.6Lf}", F(1.234567e-4));
  test("0.001235", en_US, "{:.6Lf}", F(1.234567e-3));
  test("0.012346", en_US, "{:.6Lf}", F(1.234567e-2));
  test("0.123457", en_US, "{:.6Lf}", F(1.234567e-1));
  test("1.234567", en_US, "{:.6Lf}", F(1.234567e0));
  test("12.345670", en_US, "{:.6Lf}", F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("123.456700", en_US, "{:.6Lf}", F(1.234567e2));
    test("1,234.567000", en_US, "{:.6Lf}", F(1.234567e3));
    test("12,345.670000", en_US, "{:.6Lf}", F(1.234567e4));
    test("123,456.700000", en_US, "{:.6Lf}", F(1.234567e5));
    test("1,234,567.000000", en_US, "{:.6Lf}", F(1.234567e6));
    test("12,345,670.000000", en_US, "{:.6Lf}", F(1.234567e7));
    test("123,456,700,000,000,000,000.000000", en_US, "{:.6Lf}", F(1.234567e20));
  }
  test("-0.000001", en_US, "{:.6Lf}", F(-1.234567e-6));
  test("-0.000012", en_US, "{:.6Lf}", F(-1.234567e-5));
  test("-0.000123", en_US, "{:.6Lf}", F(-1.234567e-4));
  test("-0.001235", en_US, "{:.6Lf}", F(-1.234567e-3));
  test("-0.012346", en_US, "{:.6Lf}", F(-1.234567e-2));
  test("-0.123457", en_US, "{:.6Lf}", F(-1.234567e-1));
  test("-1.234567", en_US, "{:.6Lf}", F(-1.234567e0));
  test("-12.345670", en_US, "{:.6Lf}", F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("-123.456700", en_US, "{:.6Lf}", F(-1.234567e2));
    test("-1,234.567000", en_US, "{:.6Lf}", F(-1.234567e3));
    test("-12,345.670000", en_US, "{:.6Lf}", F(-1.234567e4));
    test("-123,456.700000", en_US, "{:.6Lf}", F(-1.234567e5));
    test("-1,234,567.000000", en_US, "{:.6Lf}", F(-1.234567e6));
    test("-12,345,670.000000", en_US, "{:.6Lf}", F(-1.234567e7));
    test("-123,456,700,000,000,000,000.000000", en_US, "{:.6Lf}", F(-1.234567e20));
  }

  std::locale::global(en_US);
  test("0#000001", loc, "{:.6Lf}", F(1.234567e-6));
  test("0#000012", loc, "{:.6Lf}", F(1.234567e-5));
  test("0#000123", loc, "{:.6Lf}", F(1.234567e-4));
  test("0#001235", loc, "{:.6Lf}", F(1.234567e-3));
  test("0#012346", loc, "{:.6Lf}", F(1.234567e-2));
  test("0#123457", loc, "{:.6Lf}", F(1.234567e-1));
  test("1#234567", loc, "{:.6Lf}", F(1.234567e0));
  test("1_2#345670", loc, "{:.6Lf}", F(1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("12_3#456700", loc, "{:.6Lf}", F(1.234567e2));
    test("1_23_4#567000", loc, "{:.6Lf}", F(1.234567e3));
    test("12_34_5#670000", loc, "{:.6Lf}", F(1.234567e4));
    test("123_45_6#700000", loc, "{:.6Lf}", F(1.234567e5));
    test("1_234_56_7#000000", loc, "{:.6Lf}", F(1.234567e6));
    test("12_345_67_0#000000", loc, "{:.6Lf}", F(1.234567e7));
    test("1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000", loc, "{:.6Lf}", F(1.234567e20));
  }
  test("-0#000001", loc, "{:.6Lf}", F(-1.234567e-6));
  test("-0#000012", loc, "{:.6Lf}", F(-1.234567e-5));
  test("-0#000123", loc, "{:.6Lf}", F(-1.234567e-4));
  test("-0#001235", loc, "{:.6Lf}", F(-1.234567e-3));
  test("-0#012346", loc, "{:.6Lf}", F(-1.234567e-2));
  test("-0#123457", loc, "{:.6Lf}", F(-1.234567e-1));
  test("-1#234567", loc, "{:.6Lf}", F(-1.234567e0));
  test("-1_2#345670", loc, "{:.6Lf}", F(-1.234567e1));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("-12_3#456700", loc, "{:.6Lf}", F(-1.234567e2));
    test("-1_23_4#567000", loc, "{:.6Lf}", F(-1.234567e3));
    test("-12_34_5#670000", loc, "{:.6Lf}", F(-1.234567e4));
    test("-123_45_6#700000", loc, "{:.6Lf}", F(-1.234567e5));
    test("-1_234_56_7#000000", loc, "{:.6Lf}", F(-1.234567e6));
    test("-12_345_67_0#000000", loc, "{:.6Lf}", F(-1.234567e7));
    test("-1_2_3_4_5_6_7_0_0_0_0_0_0_00_000_00_0#000000", loc, "{:.6Lf}", F(-1.234567e20));
  }

  // *** Fill, align, zero padding ***
  if constexpr (sizeof(F) > sizeof(float)) {
    std::locale::global(en_US);
    test("1,234.567000$$$", "{:$<15.6Lf}", F(1.234567e3));
    test("$$$1,234.567000", "{:$>15.6Lf}", F(1.234567e3));
    test("$1,234.567000$$", "{:$^15.6Lf}", F(1.234567e3));
    test("0001,234.567000", "{:015.6Lf}", F(1.234567e3));
    test("-1,234.567000$$$", "{:$<16.6Lf}", F(-1.234567e3));
    test("$$$-1,234.567000", "{:$>16.6Lf}", F(-1.234567e3));
    test("$-1,234.567000$$", "{:$^16.6Lf}", F(-1.234567e3));
    test("-0001,234.567000", "{:016.6Lf}", F(-1.234567e3));

    std::locale::global(loc);
    test("1_23_4#567000$$$", "{:$<16.6Lf}", F(1.234567e3));
    test("$$$1_23_4#567000", "{:$>16.6Lf}", F(1.234567e3));
    test("$1_23_4#567000$$", "{:$^16.6Lf}", F(1.234567e3));
    test("0001_23_4#567000", "{:016.6Lf}", F(1.234567e3));
    test("-1_23_4#567000$$$", "{:$<17.6Lf}", F(-1.234567e3));
    test("$$$-1_23_4#567000", "{:$>17.6Lf}", F(-1.234567e3));
    test("$-1_23_4#567000$$", "{:$^17.6Lf}", F(-1.234567e3));
    test("-0001_23_4#567000", "{:017.6Lf}", F(-1.234567e3));

    test("1,234.567000$$$", en_US, "{:$<15.6Lf}", F(1.234567e3));
    test("$$$1,234.567000", en_US, "{:$>15.6Lf}", F(1.234567e3));
    test("$1,234.567000$$", en_US, "{:$^15.6Lf}", F(1.234567e3));
    test("0001,234.567000", en_US, "{:015.6Lf}", F(1.234567e3));
    test("-1,234.567000$$$", en_US, "{:$<16.6Lf}", F(-1.234567e3));
    test("$$$-1,234.567000", en_US, "{:$>16.6Lf}", F(-1.234567e3));
    test("$-1,234.567000$$", en_US, "{:$^16.6Lf}", F(-1.234567e3));
    test("-0001,234.567000", en_US, "{:016.6Lf}", F(-1.234567e3));

    std::locale::global(en_US);
    test("1_23_4#567000$$$", loc, "{:$<16.6Lf}", F(1.234567e3));
    test("$$$1_23_4#567000", loc, "{:$>16.6Lf}", F(1.234567e3));
    test("$1_23_4#567000$$", loc, "{:$^16.6Lf}", F(1.234567e3));
    test("0001_23_4#567000", loc, "{:016.6Lf}", F(1.234567e3));
    test("-1_23_4#567000$$$", loc, "{:$<17.6Lf}", F(-1.234567e3));
    test("$$$-1_23_4#567000", loc, "{:$>17.6Lf}", F(-1.234567e3));
    test("$-1_23_4#567000$$", loc, "{:$^17.6Lf}", F(-1.234567e3));
    test("-0001_23_4#567000", loc, "{:017.6Lf}", F(-1.234567e3));
  }
}

template <class F>
static void test_floating_point_general_lower_case() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test("1.23457e-06", "{:.6Lg}", F(1.234567e-6));
  test("1.23457e-05", "{:.6Lg}", F(1.234567e-5));
  test("0.000123457", "{:.6Lg}", F(1.234567e-4));
  test("0.00123457", "{:.6Lg}", F(1.234567e-3));
  test("0.0123457", "{:.6Lg}", F(1.234567e-2));
  test("0.123457", "{:.6Lg}", F(1.234567e-1));
  test("1.23457", "{:.6Lg}", F(1.234567e0));
  test("12.3457", "{:.6Lg}", F(1.234567e1));
  test("123.457", "{:.6Lg}", F(1.234567e2));
  test("1,234.57", "{:.6Lg}", F(1.234567e3));
  test("12,345.7", "{:.6Lg}", F(1.234567e4));
  test("123,457", "{:.6Lg}", F(1.234567e5));
  test("1.23457e+06", "{:.6Lg}", F(1.234567e6));
  test("1.23457e+07", "{:.6Lg}", F(1.234567e7));
  test("-1.23457e-06", "{:.6Lg}", F(-1.234567e-6));
  test("-1.23457e-05", "{:.6Lg}", F(-1.234567e-5));
  test("-0.000123457", "{:.6Lg}", F(-1.234567e-4));
  test("-0.00123457", "{:.6Lg}", F(-1.234567e-3));
  test("-0.0123457", "{:.6Lg}", F(-1.234567e-2));
  test("-0.123457", "{:.6Lg}", F(-1.234567e-1));
  test("-1.23457", "{:.6Lg}", F(-1.234567e0));
  test("-12.3457", "{:.6Lg}", F(-1.234567e1));
  test("-123.457", "{:.6Lg}", F(-1.234567e2));
  test("-1,234.57", "{:.6Lg}", F(-1.234567e3));
  test("-12,345.7", "{:.6Lg}", F(-1.234567e4));
  test("-123,457", "{:.6Lg}", F(-1.234567e5));
  test("-1.23457e+06", "{:.6Lg}", F(-1.234567e6));
  test("-1.23457e+07", "{:.6Lg}", F(-1.234567e7));

  std::locale::global(loc);
  test("1#23457e-06", "{:.6Lg}", F(1.234567e-6));
  test("1#23457e-05", "{:.6Lg}", F(1.234567e-5));
  test("0#000123457", "{:.6Lg}", F(1.234567e-4));
  test("0#00123457", "{:.6Lg}", F(1.234567e-3));
  test("0#0123457", "{:.6Lg}", F(1.234567e-2));
  test("0#123457", "{:.6Lg}", F(1.234567e-1));
  test("1#23457", "{:.6Lg}", F(1.234567e0));
  test("1_2#3457", "{:.6Lg}", F(1.234567e1));
  test("12_3#457", "{:.6Lg}", F(1.234567e2));
  test("1_23_4#57", "{:.6Lg}", F(1.234567e3));
  test("12_34_5#7", "{:.6Lg}", F(1.234567e4));
  test("123_45_7", "{:.6Lg}", F(1.234567e5));
  test("1#23457e+06", "{:.6Lg}", F(1.234567e6));
  test("1#23457e+07", "{:.6Lg}", F(1.234567e7));
  test("-1#23457e-06", "{:.6Lg}", F(-1.234567e-6));
  test("-1#23457e-05", "{:.6Lg}", F(-1.234567e-5));
  test("-0#000123457", "{:.6Lg}", F(-1.234567e-4));
  test("-0#00123457", "{:.6Lg}", F(-1.234567e-3));
  test("-0#0123457", "{:.6Lg}", F(-1.234567e-2));
  test("-0#123457", "{:.6Lg}", F(-1.234567e-1));
  test("-1#23457", "{:.6Lg}", F(-1.234567e0));
  test("-1_2#3457", "{:.6Lg}", F(-1.234567e1));
  test("-12_3#457", "{:.6Lg}", F(-1.234567e2));
  test("-1_23_4#57", "{:.6Lg}", F(-1.234567e3));
  test("-12_34_5#7", "{:.6Lg}", F(-1.234567e4));
  test("-123_45_7", "{:.6Lg}", F(-1.234567e5));
  test("-1#23457e+06", "{:.6Lg}", F(-1.234567e6));
  test("-1#23457e+07", "{:.6Lg}", F(-1.234567e7));

  test("1.23457e-06", en_US, "{:.6Lg}", F(1.234567e-6));
  test("1.23457e-05", en_US, "{:.6Lg}", F(1.234567e-5));
  test("0.000123457", en_US, "{:.6Lg}", F(1.234567e-4));
  test("0.00123457", en_US, "{:.6Lg}", F(1.234567e-3));
  test("0.0123457", en_US, "{:.6Lg}", F(1.234567e-2));
  test("0.123457", en_US, "{:.6Lg}", F(1.234567e-1));
  test("1.23457", en_US, "{:.6Lg}", F(1.234567e0));
  test("12.3457", en_US, "{:.6Lg}", F(1.234567e1));
  test("123.457", en_US, "{:.6Lg}", F(1.234567e2));
  test("1,234.57", en_US, "{:.6Lg}", F(1.234567e3));
  test("12,345.7", en_US, "{:.6Lg}", F(1.234567e4));
  test("123,457", en_US, "{:.6Lg}", F(1.234567e5));
  test("1.23457e+06", en_US, "{:.6Lg}", F(1.234567e6));
  test("1.23457e+07", en_US, "{:.6Lg}", F(1.234567e7));
  test("-1.23457e-06", en_US, "{:.6Lg}", F(-1.234567e-6));
  test("-1.23457e-05", en_US, "{:.6Lg}", F(-1.234567e-5));
  test("-0.000123457", en_US, "{:.6Lg}", F(-1.234567e-4));
  test("-0.00123457", en_US, "{:.6Lg}", F(-1.234567e-3));
  test("-0.0123457", en_US, "{:.6Lg}", F(-1.234567e-2));
  test("-0.123457", en_US, "{:.6Lg}", F(-1.234567e-1));
  test("-1.23457", en_US, "{:.6Lg}", F(-1.234567e0));
  test("-12.3457", en_US, "{:.6Lg}", F(-1.234567e1));
  test("-123.457", en_US, "{:.6Lg}", F(-1.234567e2));
  test("-1,234.57", en_US, "{:.6Lg}", F(-1.234567e3));
  test("-12,345.7", en_US, "{:.6Lg}", F(-1.234567e4));
  test("-123,457", en_US, "{:.6Lg}", F(-1.234567e5));
  test("-1.23457e+06", en_US, "{:.6Lg}", F(-1.234567e6));
  test("-1.23457e+07", en_US, "{:.6Lg}", F(-1.234567e7));

  std::locale::global(en_US);
  test("1#23457e-06", loc, "{:.6Lg}", F(1.234567e-6));
  test("1#23457e-05", loc, "{:.6Lg}", F(1.234567e-5));
  test("0#000123457", loc, "{:.6Lg}", F(1.234567e-4));
  test("0#00123457", loc, "{:.6Lg}", F(1.234567e-3));
  test("0#0123457", loc, "{:.6Lg}", F(1.234567e-2));
  test("0#123457", loc, "{:.6Lg}", F(1.234567e-1));
  test("1#23457", loc, "{:.6Lg}", F(1.234567e0));
  test("1_2#3457", loc, "{:.6Lg}", F(1.234567e1));
  test("12_3#457", loc, "{:.6Lg}", F(1.234567e2));
  test("1_23_4#57", loc, "{:.6Lg}", F(1.234567e3));
  test("12_34_5#7", loc, "{:.6Lg}", F(1.234567e4));
  test("123_45_7", loc, "{:.6Lg}", F(1.234567e5));
  test("1#23457e+06", loc, "{:.6Lg}", F(1.234567e6));
  test("1#23457e+07", loc, "{:.6Lg}", F(1.234567e7));
  test("-1#23457e-06", loc, "{:.6Lg}", F(-1.234567e-6));
  test("-1#23457e-05", loc, "{:.6Lg}", F(-1.234567e-5));
  test("-0#000123457", loc, "{:.6Lg}", F(-1.234567e-4));
  test("-0#00123457", loc, "{:.6Lg}", F(-1.234567e-3));
  test("-0#0123457", loc, "{:.6Lg}", F(-1.234567e-2));
  test("-0#123457", loc, "{:.6Lg}", F(-1.234567e-1));
  test("-1#23457", loc, "{:.6Lg}", F(-1.234567e0));
  test("-1_2#3457", loc, "{:.6Lg}", F(-1.234567e1));
  test("-12_3#457", loc, "{:.6Lg}", F(-1.234567e2));
  test("-1_23_4#57", loc, "{:.6Lg}", F(-1.234567e3));
  test("-12_34_5#7", loc, "{:.6Lg}", F(-1.234567e4));
  test("-123_45_7", loc, "{:.6Lg}", F(-1.234567e5));
  test("-1#23457e+06", loc, "{:.6Lg}", F(-1.234567e6));
  test("-1#23457e+07", loc, "{:.6Lg}", F(-1.234567e7));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test("1,234.57$$$", "{:$<11.6Lg}", F(1.234567e3));
  test("$$$1,234.57", "{:$>11.6Lg}", F(1.234567e3));
  test("$1,234.57$$", "{:$^11.6Lg}", F(1.234567e3));
  test("0001,234.57", "{:011.6Lg}", F(1.234567e3));
  test("-1,234.57$$$", "{:$<12.6Lg}", F(-1.234567e3));
  test("$$$-1,234.57", "{:$>12.6Lg}", F(-1.234567e3));
  test("$-1,234.57$$", "{:$^12.6Lg}", F(-1.234567e3));
  test("-0001,234.57", "{:012.6Lg}", F(-1.234567e3));

  std::locale::global(loc);
  test("1_23_4#57$$$", "{:$<12.6Lg}", F(1.234567e3));
  test("$$$1_23_4#57", "{:$>12.6Lg}", F(1.234567e3));
  test("$1_23_4#57$$", "{:$^12.6Lg}", F(1.234567e3));
  test("0001_23_4#57", "{:012.6Lg}", F(1.234567e3));
  test("-1_23_4#57$$$", "{:$<13.6Lg}", F(-1.234567e3));
  test("$$$-1_23_4#57", "{:$>13.6Lg}", F(-1.234567e3));
  test("$-1_23_4#57$$", "{:$^13.6Lg}", F(-1.234567e3));
  test("-0001_23_4#57", "{:013.6Lg}", F(-1.234567e3));

  test("1,234.57$$$", en_US, "{:$<11.6Lg}", F(1.234567e3));
  test("$$$1,234.57", en_US, "{:$>11.6Lg}", F(1.234567e3));
  test("$1,234.57$$", en_US, "{:$^11.6Lg}", F(1.234567e3));
  test("0001,234.57", en_US, "{:011.6Lg}", F(1.234567e3));
  test("-1,234.57$$$", en_US, "{:$<12.6Lg}", F(-1.234567e3));
  test("$$$-1,234.57", en_US, "{:$>12.6Lg}", F(-1.234567e3));
  test("$-1,234.57$$", en_US, "{:$^12.6Lg}", F(-1.234567e3));
  test("-0001,234.57", en_US, "{:012.6Lg}", F(-1.234567e3));

  std::locale::global(en_US);
  test("1_23_4#57$$$", loc, "{:$<12.6Lg}", F(1.234567e3));
  test("$$$1_23_4#57", loc, "{:$>12.6Lg}", F(1.234567e3));
  test("$1_23_4#57$$", loc, "{:$^12.6Lg}", F(1.234567e3));
  test("0001_23_4#57", loc, "{:012.6Lg}", F(1.234567e3));
  test("-1_23_4#57$$$", loc, "{:$<13.6Lg}", F(-1.234567e3));
  test("$$$-1_23_4#57", loc, "{:$>13.6Lg}", F(-1.234567e3));
  test("$-1_23_4#57$$", loc, "{:$^13.6Lg}", F(-1.234567e3));
  test("-0001_23_4#57", loc, "{:013.6Lg}", F(-1.234567e3));
}

template <class F>
static void test_floating_point_general_upper_case() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test("1.23457E-06", "{:.6LG}", F(1.234567e-6));
  test("1.23457E-05", "{:.6LG}", F(1.234567e-5));
  test("0.000123457", "{:.6LG}", F(1.234567e-4));
  test("0.00123457", "{:.6LG}", F(1.234567e-3));
  test("0.0123457", "{:.6LG}", F(1.234567e-2));
  test("0.123457", "{:.6LG}", F(1.234567e-1));
  test("1.23457", "{:.6LG}", F(1.234567e0));
  test("12.3457", "{:.6LG}", F(1.234567e1));
  test("123.457", "{:.6LG}", F(1.234567e2));
  test("1,234.57", "{:.6LG}", F(1.234567e3));
  test("12,345.7", "{:.6LG}", F(1.234567e4));
  test("123,457", "{:.6LG}", F(1.234567e5));
  test("1.23457E+06", "{:.6LG}", F(1.234567e6));
  test("1.23457E+07", "{:.6LG}", F(1.234567e7));
  test("-1.23457E-06", "{:.6LG}", F(-1.234567e-6));
  test("-1.23457E-05", "{:.6LG}", F(-1.234567e-5));
  test("-0.000123457", "{:.6LG}", F(-1.234567e-4));
  test("-0.00123457", "{:.6LG}", F(-1.234567e-3));
  test("-0.0123457", "{:.6LG}", F(-1.234567e-2));
  test("-0.123457", "{:.6LG}", F(-1.234567e-1));
  test("-1.23457", "{:.6LG}", F(-1.234567e0));
  test("-12.3457", "{:.6LG}", F(-1.234567e1));
  test("-123.457", "{:.6LG}", F(-1.234567e2));
  test("-1,234.57", "{:.6LG}", F(-1.234567e3));
  test("-12,345.7", "{:.6LG}", F(-1.234567e4));
  test("-123,457", "{:.6LG}", F(-1.234567e5));
  test("-1.23457E+06", "{:.6LG}", F(-1.234567e6));
  test("-1.23457E+07", "{:.6LG}", F(-1.234567e7));

  std::locale::global(loc);
  test("1#23457E-06", "{:.6LG}", F(1.234567e-6));
  test("1#23457E-05", "{:.6LG}", F(1.234567e-5));
  test("0#000123457", "{:.6LG}", F(1.234567e-4));
  test("0#00123457", "{:.6LG}", F(1.234567e-3));
  test("0#0123457", "{:.6LG}", F(1.234567e-2));
  test("0#123457", "{:.6LG}", F(1.234567e-1));
  test("1#23457", "{:.6LG}", F(1.234567e0));
  test("1_2#3457", "{:.6LG}", F(1.234567e1));
  test("12_3#457", "{:.6LG}", F(1.234567e2));
  test("1_23_4#57", "{:.6LG}", F(1.234567e3));
  test("12_34_5#7", "{:.6LG}", F(1.234567e4));
  test("123_45_7", "{:.6LG}", F(1.234567e5));
  test("1#23457E+06", "{:.6LG}", F(1.234567e6));
  test("1#23457E+07", "{:.6LG}", F(1.234567e7));
  test("-1#23457E-06", "{:.6LG}", F(-1.234567e-6));
  test("-1#23457E-05", "{:.6LG}", F(-1.234567e-5));
  test("-0#000123457", "{:.6LG}", F(-1.234567e-4));
  test("-0#00123457", "{:.6LG}", F(-1.234567e-3));
  test("-0#0123457", "{:.6LG}", F(-1.234567e-2));
  test("-0#123457", "{:.6LG}", F(-1.234567e-1));
  test("-1#23457", "{:.6LG}", F(-1.234567e0));
  test("-1_2#3457", "{:.6LG}", F(-1.234567e1));
  test("-12_3#457", "{:.6LG}", F(-1.234567e2));
  test("-1_23_4#57", "{:.6LG}", F(-1.234567e3));
  test("-12_34_5#7", "{:.6LG}", F(-1.234567e4));
  test("-123_45_7", "{:.6LG}", F(-1.234567e5));
  test("-1#23457E+06", "{:.6LG}", F(-1.234567e6));
  test("-1#23457E+07", "{:.6LG}", F(-1.234567e7));

  test("1.23457E-06", en_US, "{:.6LG}", F(1.234567e-6));
  test("1.23457E-05", en_US, "{:.6LG}", F(1.234567e-5));
  test("0.000123457", en_US, "{:.6LG}", F(1.234567e-4));
  test("0.00123457", en_US, "{:.6LG}", F(1.234567e-3));
  test("0.0123457", en_US, "{:.6LG}", F(1.234567e-2));
  test("0.123457", en_US, "{:.6LG}", F(1.234567e-1));
  test("1.23457", en_US, "{:.6LG}", F(1.234567e0));
  test("12.3457", en_US, "{:.6LG}", F(1.234567e1));
  test("123.457", en_US, "{:.6LG}", F(1.234567e2));
  test("1,234.57", en_US, "{:.6LG}", F(1.234567e3));
  test("12,345.7", en_US, "{:.6LG}", F(1.234567e4));
  test("123,457", en_US, "{:.6LG}", F(1.234567e5));
  test("1.23457E+06", en_US, "{:.6LG}", F(1.234567e6));
  test("1.23457E+07", en_US, "{:.6LG}", F(1.234567e7));
  test("-1.23457E-06", en_US, "{:.6LG}", F(-1.234567e-6));
  test("-1.23457E-05", en_US, "{:.6LG}", F(-1.234567e-5));
  test("-0.000123457", en_US, "{:.6LG}", F(-1.234567e-4));
  test("-0.00123457", en_US, "{:.6LG}", F(-1.234567e-3));
  test("-0.0123457", en_US, "{:.6LG}", F(-1.234567e-2));
  test("-0.123457", en_US, "{:.6LG}", F(-1.234567e-1));
  test("-1.23457", en_US, "{:.6LG}", F(-1.234567e0));
  test("-12.3457", en_US, "{:.6LG}", F(-1.234567e1));
  test("-123.457", en_US, "{:.6LG}", F(-1.234567e2));
  test("-1,234.57", en_US, "{:.6LG}", F(-1.234567e3));
  test("-12,345.7", en_US, "{:.6LG}", F(-1.234567e4));
  test("-123,457", en_US, "{:.6LG}", F(-1.234567e5));
  test("-1.23457E+06", en_US, "{:.6LG}", F(-1.234567e6));
  test("-1.23457E+07", en_US, "{:.6LG}", F(-1.234567e7));

  std::locale::global(en_US);
  test("1#23457E-06", loc, "{:.6LG}", F(1.234567e-6));
  test("1#23457E-05", loc, "{:.6LG}", F(1.234567e-5));
  test("0#000123457", loc, "{:.6LG}", F(1.234567e-4));
  test("0#00123457", loc, "{:.6LG}", F(1.234567e-3));
  test("0#0123457", loc, "{:.6LG}", F(1.234567e-2));
  test("0#123457", loc, "{:.6LG}", F(1.234567e-1));
  test("1#23457", loc, "{:.6LG}", F(1.234567e0));
  test("1_2#3457", loc, "{:.6LG}", F(1.234567e1));
  test("12_3#457", loc, "{:.6LG}", F(1.234567e2));
  test("1_23_4#57", loc, "{:.6LG}", F(1.234567e3));
  test("12_34_5#7", loc, "{:.6LG}", F(1.234567e4));
  test("123_45_7", loc, "{:.6LG}", F(1.234567e5));
  test("1#23457E+06", loc, "{:.6LG}", F(1.234567e6));
  test("1#23457E+07", loc, "{:.6LG}", F(1.234567e7));
  test("-1#23457E-06", loc, "{:.6LG}", F(-1.234567e-6));
  test("-1#23457E-05", loc, "{:.6LG}", F(-1.234567e-5));
  test("-0#000123457", loc, "{:.6LG}", F(-1.234567e-4));
  test("-0#00123457", loc, "{:.6LG}", F(-1.234567e-3));
  test("-0#0123457", loc, "{:.6LG}", F(-1.234567e-2));
  test("-0#123457", loc, "{:.6LG}", F(-1.234567e-1));
  test("-1#23457", loc, "{:.6LG}", F(-1.234567e0));
  test("-1_2#3457", loc, "{:.6LG}", F(-1.234567e1));
  test("-12_3#457", loc, "{:.6LG}", F(-1.234567e2));
  test("-1_23_4#57", loc, "{:.6LG}", F(-1.234567e3));
  test("-12_34_5#7", loc, "{:.6LG}", F(-1.234567e4));
  test("-123_45_7", loc, "{:.6LG}", F(-1.234567e5));
  test("-1#23457E+06", loc, "{:.6LG}", F(-1.234567e6));
  test("-1#23457E+07", loc, "{:.6LG}", F(-1.234567e7));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test("1,234.57$$$", "{:$<11.6LG}", F(1.234567e3));
  test("$$$1,234.57", "{:$>11.6LG}", F(1.234567e3));
  test("$1,234.57$$", "{:$^11.6LG}", F(1.234567e3));
  test("0001,234.57", "{:011.6LG}", F(1.234567e3));
  test("-1,234.57$$$", "{:$<12.6LG}", F(-1.234567e3));
  test("$$$-1,234.57", "{:$>12.6LG}", F(-1.234567e3));
  test("$-1,234.57$$", "{:$^12.6LG}", F(-1.234567e3));
  test("-0001,234.57", "{:012.6LG}", F(-1.234567e3));

  std::locale::global(loc);
  test("1_23_4#57$$$", "{:$<12.6LG}", F(1.234567e3));
  test("$$$1_23_4#57", "{:$>12.6LG}", F(1.234567e3));
  test("$1_23_4#57$$", "{:$^12.6LG}", F(1.234567e3));
  test("0001_23_4#57", "{:012.6LG}", F(1.234567e3));
  test("-1_23_4#57$$$", "{:$<13.6LG}", F(-1.234567e3));
  test("$$$-1_23_4#57", "{:$>13.6LG}", F(-1.234567e3));
  test("$-1_23_4#57$$", "{:$^13.6LG}", F(-1.234567e3));
  test("-0001_23_4#57", "{:013.6LG}", F(-1.234567e3));

  test("1,234.57$$$", en_US, "{:$<11.6LG}", F(1.234567e3));
  test("$$$1,234.57", en_US, "{:$>11.6LG}", F(1.234567e3));
  test("$1,234.57$$", en_US, "{:$^11.6LG}", F(1.234567e3));
  test("0001,234.57", en_US, "{:011.6LG}", F(1.234567e3));
  test("-1,234.57$$$", en_US, "{:$<12.6LG}", F(-1.234567e3));
  test("$$$-1,234.57", en_US, "{:$>12.6LG}", F(-1.234567e3));
  test("$-1,234.57$$", en_US, "{:$^12.6LG}", F(-1.234567e3));
  test("-0001,234.57", en_US, "{:012.6LG}", F(-1.234567e3));

  std::locale::global(en_US);
  test("1_23_4#57$$$", loc, "{:$<12.6LG}", F(1.234567e3));
  test("$$$1_23_4#57", loc, "{:$>12.6LG}", F(1.234567e3));
  test("$1_23_4#57$$", loc, "{:$^12.6LG}", F(1.234567e3));
  test("0001_23_4#57", loc, "{:012.6LG}", F(1.234567e3));
  test("-1_23_4#57$$$", loc, "{:$<13.6LG}", F(-1.234567e3));
  test("$$$-1_23_4#57", loc, "{:$>13.6LG}", F(-1.234567e3));
  test("$-1_23_4#57$$", loc, "{:$^13.6LG}", F(-1.234567e3));
  test("-0001_23_4#57", loc, "{:013.6LG}", F(-1.234567e3));
}

template <class F>
static void test_floating_point_default() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test("1.234567e-06", "{:L}", F(1.234567e-6));
  test("1.234567e-05", "{:L}", F(1.234567e-5));
  test("0.0001234567", "{:L}", F(1.234567e-4));
  test("0.001234567", "{:L}", F(1.234567e-3));
  test("0.01234567", "{:L}", F(1.234567e-2));
  test("0.1234567", "{:L}", F(1.234567e-1));
  test("1.234567", "{:L}", F(1.234567e0));
  test("12.34567", "{:L}", F(1.234567e1));
  test("123.4567", "{:L}", F(1.234567e2));
  test("1,234.567", "{:L}", F(1.234567e3));
  test("12,345.67", "{:L}", F(1.234567e4));
  test("123,456.7", "{:L}", F(1.234567e5));
  test("1,234,567", "{:L}", F(1.234567e6));
  test("12,345,670", "{:L}", F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("123,456,700", "{:L}", F(1.234567e8));
    test("1,234,567,000", "{:L}", F(1.234567e9));
    test("12,345,670,000", "{:L}", F(1.234567e10));
    test("123,456,700,000", "{:L}", F(1.234567e11));
    test("1.234567e+12", "{:L}", F(1.234567e12));
    test("1.234567e+13", "{:L}", F(1.234567e13));
  }
  test("-1.234567e-06", "{:L}", F(-1.234567e-6));
  test("-1.234567e-05", "{:L}", F(-1.234567e-5));
  test("-0.0001234567", "{:L}", F(-1.234567e-4));
  test("-0.001234567", "{:L}", F(-1.234567e-3));
  test("-0.01234567", "{:L}", F(-1.234567e-2));
  test("-0.1234567", "{:L}", F(-1.234567e-1));
  test("-1.234567", "{:L}", F(-1.234567e0));
  test("-12.34567", "{:L}", F(-1.234567e1));
  test("-123.4567", "{:L}", F(-1.234567e2));
  test("-1,234.567", "{:L}", F(-1.234567e3));
  test("-12,345.67", "{:L}", F(-1.234567e4));
  test("-123,456.7", "{:L}", F(-1.234567e5));
  test("-1,234,567", "{:L}", F(-1.234567e6));
  test("-12,345,670", "{:L}", F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("-123,456,700", "{:L}", F(-1.234567e8));
    test("-1,234,567,000", "{:L}", F(-1.234567e9));
    test("-12,345,670,000", "{:L}", F(-1.234567e10));
    test("-123,456,700,000", "{:L}", F(-1.234567e11));
    test("-1.234567e+12", "{:L}", F(-1.234567e12));
    test("-1.234567e+13", "{:L}", F(-1.234567e13));
  }

  std::locale::global(loc);
  test("1#234567e-06", "{:L}", F(1.234567e-6));
  test("1#234567e-05", "{:L}", F(1.234567e-5));
  test("0#0001234567", "{:L}", F(1.234567e-4));
  test("0#001234567", "{:L}", F(1.234567e-3));
  test("0#01234567", "{:L}", F(1.234567e-2));
  test("0#1234567", "{:L}", F(1.234567e-1));
  test("1#234567", "{:L}", F(1.234567e0));
  test("1_2#34567", "{:L}", F(1.234567e1));
  test("12_3#4567", "{:L}", F(1.234567e2));
  test("1_23_4#567", "{:L}", F(1.234567e3));
  test("12_34_5#67", "{:L}", F(1.234567e4));
  test("123_45_6#7", "{:L}", F(1.234567e5));
  test("1_234_56_7", "{:L}", F(1.234567e6));
  test("12_345_67_0", "{:L}", F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("1_23_456_70_0", "{:L}", F(1.234567e8));
    test("1_2_34_567_00_0", "{:L}", F(1.234567e9));
    test("1_2_3_45_670_00_0", "{:L}", F(1.234567e10));
    test("1_2_3_4_56_700_00_0", "{:L}", F(1.234567e11));
    test("1#234567e+12", "{:L}", F(1.234567e12));
    test("1#234567e+13", "{:L}", F(1.234567e13));
  }
  test("-1#234567e-06", "{:L}", F(-1.234567e-6));
  test("-1#234567e-05", "{:L}", F(-1.234567e-5));
  test("-0#0001234567", "{:L}", F(-1.234567e-4));
  test("-0#001234567", "{:L}", F(-1.234567e-3));
  test("-0#01234567", "{:L}", F(-1.234567e-2));
  test("-0#1234567", "{:L}", F(-1.234567e-1));
  test("-1#234567", "{:L}", F(-1.234567e0));
  test("-1_2#34567", "{:L}", F(-1.234567e1));
  test("-12_3#4567", "{:L}", F(-1.234567e2));
  test("-1_23_4#567", "{:L}", F(-1.234567e3));
  test("-12_34_5#67", "{:L}", F(-1.234567e4));
  test("-123_45_6#7", "{:L}", F(-1.234567e5));
  test("-1_234_56_7", "{:L}", F(-1.234567e6));
  test("-12_345_67_0", "{:L}", F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("-1_23_456_70_0", "{:L}", F(-1.234567e8));
    test("-1_2_34_567_00_0", "{:L}", F(-1.234567e9));
    test("-1_2_3_45_670_00_0", "{:L}", F(-1.234567e10));
    test("-1_2_3_4_56_700_00_0", "{:L}", F(-1.234567e11));
    test("-1#234567e+12", "{:L}", F(-1.234567e12));
    test("-1#234567e+13", "{:L}", F(-1.234567e13));
  }

  test("1.234567e-06", en_US, "{:L}", F(1.234567e-6));
  test("1.234567e-05", en_US, "{:L}", F(1.234567e-5));
  test("0.0001234567", en_US, "{:L}", F(1.234567e-4));
  test("0.001234567", en_US, "{:L}", F(1.234567e-3));
  test("0.01234567", en_US, "{:L}", F(1.234567e-2));
  test("0.1234567", en_US, "{:L}", F(1.234567e-1));
  test("1.234567", en_US, "{:L}", F(1.234567e0));
  test("12.34567", en_US, "{:L}", F(1.234567e1));
  test("123.4567", en_US, "{:L}", F(1.234567e2));
  test("1,234.567", en_US, "{:L}", F(1.234567e3));
  test("12,345.67", en_US, "{:L}", F(1.234567e4));
  test("123,456.7", en_US, "{:L}", F(1.234567e5));
  test("1,234,567", en_US, "{:L}", F(1.234567e6));
  test("12,345,670", en_US, "{:L}", F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("123,456,700", en_US, "{:L}", F(1.234567e8));
    test("1,234,567,000", en_US, "{:L}", F(1.234567e9));
    test("12,345,670,000", en_US, "{:L}", F(1.234567e10));
    test("123,456,700,000", en_US, "{:L}", F(1.234567e11));
    test("1.234567e+12", en_US, "{:L}", F(1.234567e12));
    test("1.234567e+13", en_US, "{:L}", F(1.234567e13));
  }
  test("-1.234567e-06", en_US, "{:L}", F(-1.234567e-6));
  test("-1.234567e-05", en_US, "{:L}", F(-1.234567e-5));
  test("-0.0001234567", en_US, "{:L}", F(-1.234567e-4));
  test("-0.001234567", en_US, "{:L}", F(-1.234567e-3));
  test("-0.01234567", en_US, "{:L}", F(-1.234567e-2));
  test("-0.1234567", en_US, "{:L}", F(-1.234567e-1));
  test("-1.234567", en_US, "{:L}", F(-1.234567e0));
  test("-12.34567", en_US, "{:L}", F(-1.234567e1));
  test("-123.4567", en_US, "{:L}", F(-1.234567e2));
  test("-1,234.567", en_US, "{:L}", F(-1.234567e3));
  test("-12,345.67", en_US, "{:L}", F(-1.234567e4));
  test("-123,456.7", en_US, "{:L}", F(-1.234567e5));
  test("-1,234,567", en_US, "{:L}", F(-1.234567e6));
  test("-12,345,670", en_US, "{:L}", F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("-123,456,700", en_US, "{:L}", F(-1.234567e8));
    test("-1,234,567,000", en_US, "{:L}", F(-1.234567e9));
    test("-12,345,670,000", en_US, "{:L}", F(-1.234567e10));
    test("-123,456,700,000", en_US, "{:L}", F(-1.234567e11));
    test("-1.234567e+12", en_US, "{:L}", F(-1.234567e12));
    test("-1.234567e+13", en_US, "{:L}", F(-1.234567e13));
  }

  std::locale::global(en_US);
  test("1#234567e-06", loc, "{:L}", F(1.234567e-6));
  test("1#234567e-05", loc, "{:L}", F(1.234567e-5));
  test("0#0001234567", loc, "{:L}", F(1.234567e-4));
  test("0#001234567", loc, "{:L}", F(1.234567e-3));
  test("0#01234567", loc, "{:L}", F(1.234567e-2));
  test("0#1234567", loc, "{:L}", F(1.234567e-1));
  test("1#234567", loc, "{:L}", F(1.234567e0));
  test("1_2#34567", loc, "{:L}", F(1.234567e1));
  test("12_3#4567", loc, "{:L}", F(1.234567e2));
  test("1_23_4#567", loc, "{:L}", F(1.234567e3));
  test("12_34_5#67", loc, "{:L}", F(1.234567e4));
  test("123_45_6#7", loc, "{:L}", F(1.234567e5));
  test("1_234_56_7", loc, "{:L}", F(1.234567e6));
  test("12_345_67_0", loc, "{:L}", F(1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("1_23_456_70_0", loc, "{:L}", F(1.234567e8));
    test("1_2_34_567_00_0", loc, "{:L}", F(1.234567e9));
    test("1_2_3_45_670_00_0", loc, "{:L}", F(1.234567e10));
    test("1_2_3_4_56_700_00_0", loc, "{:L}", F(1.234567e11));
    test("1#234567e+12", loc, "{:L}", F(1.234567e12));
    test("1#234567e+13", loc, "{:L}", F(1.234567e13));
  }
  test("-1#234567e-06", loc, "{:L}", F(-1.234567e-6));
  test("-1#234567e-05", loc, "{:L}", F(-1.234567e-5));
  test("-0#0001234567", loc, "{:L}", F(-1.234567e-4));
  test("-0#001234567", loc, "{:L}", F(-1.234567e-3));
  test("-0#01234567", loc, "{:L}", F(-1.234567e-2));
  test("-0#1234567", loc, "{:L}", F(-1.234567e-1));
  test("-1#234567", loc, "{:L}", F(-1.234567e0));
  test("-1_2#34567", loc, "{:L}", F(-1.234567e1));
  test("-12_3#4567", loc, "{:L}", F(-1.234567e2));
  test("-1_23_4#567", loc, "{:L}", F(-1.234567e3));
  test("-12_34_5#67", loc, "{:L}", F(-1.234567e4));
  test("-123_45_6#7", loc, "{:L}", F(-1.234567e5));
  test("-1_234_56_7", loc, "{:L}", F(-1.234567e6));
  test("-12_345_67_0", loc, "{:L}", F(-1.234567e7));
  if constexpr (sizeof(F) > sizeof(float)) {
    test("-1_23_456_70_0", loc, "{:L}", F(-1.234567e8));
    test("-1_2_34_567_00_0", loc, "{:L}", F(-1.234567e9));
    test("-1_2_3_45_670_00_0", loc, "{:L}", F(-1.234567e10));
    test("-1_2_3_4_56_700_00_0", loc, "{:L}", F(-1.234567e11));
    test("-1#234567e+12", loc, "{:L}", F(-1.234567e12));
    test("-1#234567e+13", loc, "{:L}", F(-1.234567e13));
  }

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test("1,234.567$$$", "{:$<12L}", F(1.234567e3));
  test("$$$1,234.567", "{:$>12L}", F(1.234567e3));
  test("$1,234.567$$", "{:$^12L}", F(1.234567e3));
  test("0001,234.567", "{:012L}", F(1.234567e3));
  test("-1,234.567$$$", "{:$<13L}", F(-1.234567e3));
  test("$$$-1,234.567", "{:$>13L}", F(-1.234567e3));
  test("$-1,234.567$$", "{:$^13L}", F(-1.234567e3));
  test("-0001,234.567", "{:013L}", F(-1.234567e3));

  std::locale::global(loc);
  test("1_23_4#567$$$", "{:$<13L}", F(1.234567e3));
  test("$$$1_23_4#567", "{:$>13L}", F(1.234567e3));
  test("$1_23_4#567$$", "{:$^13L}", F(1.234567e3));
  test("0001_23_4#567", "{:013L}", F(1.234567e3));
  test("-1_23_4#567$$$", "{:$<14L}", F(-1.234567e3));
  test("$$$-1_23_4#567", "{:$>14L}", F(-1.234567e3));
  test("$-1_23_4#567$$", "{:$^14L}", F(-1.234567e3));
  test("-0001_23_4#567", "{:014L}", F(-1.234567e3));

  test("1,234.567$$$", en_US, "{:$<12L}", F(1.234567e3));
  test("$$$1,234.567", en_US, "{:$>12L}", F(1.234567e3));
  test("$1,234.567$$", en_US, "{:$^12L}", F(1.234567e3));
  test("0001,234.567", en_US, "{:012L}", F(1.234567e3));
  test("-1,234.567$$$", en_US, "{:$<13L}", F(-1.234567e3));
  test("$$$-1,234.567", en_US, "{:$>13L}", F(-1.234567e3));
  test("$-1,234.567$$", en_US, "{:$^13L}", F(-1.234567e3));
  test("-0001,234.567", en_US, "{:013L}", F(-1.234567e3));

  std::locale::global(en_US);
  test("1_23_4#567$$$", loc, "{:$<13L}", F(1.234567e3));
  test("$$$1_23_4#567", loc, "{:$>13L}", F(1.234567e3));
  test("$1_23_4#567$$", loc, "{:$^13L}", F(1.234567e3));
  test("0001_23_4#567", loc, "{:013L}", F(1.234567e3));
  test("-1_23_4#567$$$", loc, "{:$<14L}", F(-1.234567e3));
  test("$$$-1_23_4#567", loc, "{:$>14L}", F(-1.234567e3));
  test("$-1_23_4#567$$", loc, "{:$^14L}", F(-1.234567e3));
  test("-0001_23_4#567", loc, "{:014L}", F(-1.234567e3));
}

template <class F>
static void test_floating_point_default_precision() {
  std::locale loc   = std::locale(std::locale(), new numpunct<char>());
  std::locale en_US = std::locale(LOCALE_en_US_UTF_8);

  // *** Basic ***
  std::locale::global(en_US);
  test("1.23457e-06", "{:.6L}", F(1.234567e-6));
  test("1.23457e-05", "{:.6L}", F(1.234567e-5));
  test("0.000123457", "{:.6L}", F(1.234567e-4));
  test("0.00123457", "{:.6L}", F(1.234567e-3));
  test("0.0123457", "{:.6L}", F(1.234567e-2));
  test("0.123457", "{:.6L}", F(1.234567e-1));
  test("1.23457", "{:.6L}", F(1.234567e0));
  test("12.3457", "{:.6L}", F(1.234567e1));
  test("123.457", "{:.6L}", F(1.234567e2));
  test("1,234.57", "{:.6L}", F(1.234567e3));
  test("12,345.7", "{:.6L}", F(1.234567e4));
  test("123,457", "{:.6L}", F(1.234567e5));
  test("1.23457e+06", "{:.6L}", F(1.234567e6));
  test("1.23457e+07", "{:.6L}", F(1.234567e7));
  test("-1.23457e-06", "{:.6L}", F(-1.234567e-6));
  test("-1.23457e-05", "{:.6L}", F(-1.234567e-5));
  test("-0.000123457", "{:.6L}", F(-1.234567e-4));
  test("-0.00123457", "{:.6L}", F(-1.234567e-3));
  test("-0.0123457", "{:.6L}", F(-1.234567e-2));
  test("-0.123457", "{:.6L}", F(-1.234567e-1));
  test("-1.23457", "{:.6L}", F(-1.234567e0));
  test("-12.3457", "{:.6L}", F(-1.234567e1));
  test("-123.457", "{:.6L}", F(-1.234567e2));
  test("-1,234.57", "{:.6L}", F(-1.234567e3));
  test("-12,345.7", "{:.6L}", F(-1.234567e4));
  test("-123,457", "{:.6L}", F(-1.234567e5));
  test("-1.23457e+06", "{:.6L}", F(-1.234567e6));
  test("-1.23457e+07", "{:.6L}", F(-1.234567e7));

  std::locale::global(loc);
  test("1#23457e-06", "{:.6L}", F(1.234567e-6));
  test("1#23457e-05", "{:.6L}", F(1.234567e-5));
  test("0#000123457", "{:.6L}", F(1.234567e-4));
  test("0#00123457", "{:.6L}", F(1.234567e-3));
  test("0#0123457", "{:.6L}", F(1.234567e-2));
  test("0#123457", "{:.6L}", F(1.234567e-1));
  test("1#23457", "{:.6L}", F(1.234567e0));
  test("1_2#3457", "{:.6L}", F(1.234567e1));
  test("12_3#457", "{:.6L}", F(1.234567e2));
  test("1_23_4#57", "{:.6L}", F(1.234567e3));
  test("12_34_5#7", "{:.6L}", F(1.234567e4));
  test("123_45_7", "{:.6L}", F(1.234567e5));
  test("1#23457e+06", "{:.6L}", F(1.234567e6));
  test("1#23457e+07", "{:.6L}", F(1.234567e7));
  test("-1#23457e-06", "{:.6L}", F(-1.234567e-6));
  test("-1#23457e-05", "{:.6L}", F(-1.234567e-5));
  test("-0#000123457", "{:.6L}", F(-1.234567e-4));
  test("-0#00123457", "{:.6L}", F(-1.234567e-3));
  test("-0#0123457", "{:.6L}", F(-1.234567e-2));
  test("-0#123457", "{:.6L}", F(-1.234567e-1));
  test("-1#23457", "{:.6L}", F(-1.234567e0));
  test("-1_2#3457", "{:.6L}", F(-1.234567e1));
  test("-12_3#457", "{:.6L}", F(-1.234567e2));
  test("-1_23_4#57", "{:.6L}", F(-1.234567e3));
  test("-12_34_5#7", "{:.6L}", F(-1.234567e4));
  test("-123_45_7", "{:.6L}", F(-1.234567e5));
  test("-1#23457e+06", "{:.6L}", F(-1.234567e6));
  test("-1#23457e+07", "{:.6L}", F(-1.234567e7));

  test("1.23457e-06", en_US, "{:.6L}", F(1.234567e-6));
  test("1.23457e-05", en_US, "{:.6L}", F(1.234567e-5));
  test("0.000123457", en_US, "{:.6L}", F(1.234567e-4));
  test("0.00123457", en_US, "{:.6L}", F(1.234567e-3));
  test("0.0123457", en_US, "{:.6L}", F(1.234567e-2));
  test("0.123457", en_US, "{:.6L}", F(1.234567e-1));
  test("1.23457", en_US, "{:.6L}", F(1.234567e0));
  test("12.3457", en_US, "{:.6L}", F(1.234567e1));
  test("123.457", en_US, "{:.6L}", F(1.234567e2));
  test("1,234.57", en_US, "{:.6L}", F(1.234567e3));
  test("12,345.7", en_US, "{:.6L}", F(1.234567e4));
  test("123,457", en_US, "{:.6L}", F(1.234567e5));
  test("1.23457e+06", en_US, "{:.6L}", F(1.234567e6));
  test("1.23457e+07", en_US, "{:.6L}", F(1.234567e7));
  test("-1.23457e-06", en_US, "{:.6L}", F(-1.234567e-6));
  test("-1.23457e-05", en_US, "{:.6L}", F(-1.234567e-5));
  test("-0.000123457", en_US, "{:.6L}", F(-1.234567e-4));
  test("-0.00123457", en_US, "{:.6L}", F(-1.234567e-3));
  test("-0.0123457", en_US, "{:.6L}", F(-1.234567e-2));
  test("-0.123457", en_US, "{:.6L}", F(-1.234567e-1));
  test("-1.23457", en_US, "{:.6L}", F(-1.234567e0));
  test("-12.3457", en_US, "{:.6L}", F(-1.234567e1));
  test("-123.457", en_US, "{:.6L}", F(-1.234567e2));
  test("-1,234.57", en_US, "{:.6L}", F(-1.234567e3));
  test("-12,345.7", en_US, "{:.6L}", F(-1.234567e4));
  test("-123,457", en_US, "{:.6L}", F(-1.234567e5));
  test("-1.23457e+06", en_US, "{:.6L}", F(-1.234567e6));
  test("-1.23457e+07", en_US, "{:.6L}", F(-1.234567e7));

  std::locale::global(en_US);
  test("1#23457e-06", loc, "{:.6L}", F(1.234567e-6));
  test("1#23457e-05", loc, "{:.6L}", F(1.234567e-5));
  test("0#000123457", loc, "{:.6L}", F(1.234567e-4));
  test("0#00123457", loc, "{:.6L}", F(1.234567e-3));
  test("0#0123457", loc, "{:.6L}", F(1.234567e-2));
  test("0#123457", loc, "{:.6L}", F(1.234567e-1));
  test("1#23457", loc, "{:.6L}", F(1.234567e0));
  test("1_2#3457", loc, "{:.6L}", F(1.234567e1));
  test("12_3#457", loc, "{:.6L}", F(1.234567e2));
  test("1_23_4#57", loc, "{:.6L}", F(1.234567e3));
  test("12_34_5#7", loc, "{:.6L}", F(1.234567e4));
  test("123_45_7", loc, "{:.6L}", F(1.234567e5));
  test("1#23457e+06", loc, "{:.6L}", F(1.234567e6));
  test("1#23457e+07", loc, "{:.6L}", F(1.234567e7));
  test("-1#23457e-06", loc, "{:.6L}", F(-1.234567e-6));
  test("-1#23457e-05", loc, "{:.6L}", F(-1.234567e-5));
  test("-0#000123457", loc, "{:.6L}", F(-1.234567e-4));
  test("-0#00123457", loc, "{:.6L}", F(-1.234567e-3));
  test("-0#0123457", loc, "{:.6L}", F(-1.234567e-2));
  test("-0#123457", loc, "{:.6L}", F(-1.234567e-1));
  test("-1#23457", loc, "{:.6L}", F(-1.234567e0));
  test("-1_2#3457", loc, "{:.6L}", F(-1.234567e1));
  test("-12_3#457", loc, "{:.6L}", F(-1.234567e2));
  test("-1_23_4#57", loc, "{:.6L}", F(-1.234567e3));
  test("-12_34_5#7", loc, "{:.6L}", F(-1.234567e4));
  test("-123_45_7", loc, "{:.6L}", F(-1.234567e5));
  test("-1#23457e+06", loc, "{:.6L}", F(-1.234567e6));
  test("-1#23457e+07", loc, "{:.6L}", F(-1.234567e7));

  // *** Fill, align, zero padding ***
  std::locale::global(en_US);
  test("1,234.57$$$", "{:$<11.6L}", F(1.234567e3));
  test("$$$1,234.57", "{:$>11.6L}", F(1.234567e3));
  test("$1,234.57$$", "{:$^11.6L}", F(1.234567e3));
  test("0001,234.57", "{:011.6L}", F(1.234567e3));
  test("-1,234.57$$$", "{:$<12.6L}", F(-1.234567e3));
  test("$$$-1,234.57", "{:$>12.6L}", F(-1.234567e3));
  test("$-1,234.57$$", "{:$^12.6L}", F(-1.234567e3));
  test("-0001,234.57", "{:012.6L}", F(-1.234567e3));

  std::locale::global(loc);
  test("1_23_4#57$$$", "{:$<12.6L}", F(1.234567e3));
  test("$$$1_23_4#57", "{:$>12.6L}", F(1.234567e3));
  test("$1_23_4#57$$", "{:$^12.6L}", F(1.234567e3));
  test("0001_23_4#57", "{:012.6L}", F(1.234567e3));
  test("-1_23_4#57$$$", "{:$<13.6L}", F(-1.234567e3));
  test("$$$-1_23_4#57", "{:$>13.6L}", F(-1.234567e3));
  test("$-1_23_4#57$$", "{:$^13.6L}", F(-1.234567e3));
  test("-0001_23_4#57", "{:013.6L}", F(-1.234567e3));

  test("1,234.57$$$", en_US, "{:$<11.6L}", F(1.234567e3));
  test("$$$1,234.57", en_US, "{:$>11.6L}", F(1.234567e3));
  test("$1,234.57$$", en_US, "{:$^11.6L}", F(1.234567e3));
  test("0001,234.57", en_US, "{:011.6L}", F(1.234567e3));
  test("-1,234.57$$$", en_US, "{:$<12.6L}", F(-1.234567e3));
  test("$$$-1,234.57", en_US, "{:$>12.6L}", F(-1.234567e3));
  test("$-1,234.57$$", en_US, "{:$^12.6L}", F(-1.234567e3));
  test("-0001,234.57", en_US, "{:012.6L}", F(-1.234567e3));

  std::locale::global(en_US);
  test("1_23_4#57$$$", loc, "{:$<12.6L}", F(1.234567e3));
  test("$$$1_23_4#57", loc, "{:$>12.6L}", F(1.234567e3));
  test("$1_23_4#57$$", loc, "{:$^12.6L}", F(1.234567e3));
  test("0001_23_4#57", loc, "{:012.6L}", F(1.234567e3));
  test("-1_23_4#57$$$", loc, "{:$<13.6L}", F(-1.234567e3));
  test("$$$-1_23_4#57", loc, "{:$>13.6L}", F(-1.234567e3));
  test("$-1_23_4#57$$", loc, "{:$^13.6L}", F(-1.234567e3));
  test("-0001_23_4#57", loc, "{:013.6L}", F(-1.234567e3));
}

template <class F>
static void test_floating_point() {
  test_floating_point_hex_lower_case<F>();
  test_floating_point_hex_upper_case<F>();
  test_floating_point_hex_lower_case_precision<F>();
  test_floating_point_hex_upper_case_precision<F>();

  test_floating_point_scientific_lower_case<F>();
  test_floating_point_scientific_upper_case<F>();

  test_floating_point_fixed_lower_case<F>();
  test_floating_point_fixed_upper_case<F>();

  test_floating_point_general_lower_case<F>();
  test_floating_point_general_upper_case<F>();

  test_floating_point_default<F>();
  test_floating_point_default_precision<F>();
}

int main(int, char**) {
  test_bool();
  test_integer();
  test_floating_point<float>();
  test_floating_point<double>();
  test_floating_point<long double>();

  return 0;
}
