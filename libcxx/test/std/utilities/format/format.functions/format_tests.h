//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_UTILITIES_FORMAT_FORMAT_FUNCTIONS_FORMAT_TESTS_H
#define TEST_STD_UTILITIES_FORMAT_FORMAT_FUNCTIONS_FORMAT_TESTS_H

#include <format>

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iterator>

#include "make_string.h"
#include "string_literal.h"
#include "test_macros.h"

// In this file the following template types are used:
// TestFunction must be callable as check(expected-result, string-to-format, args-to-format...)
// ExceptionTest must be callable as check_exception(expected-exception, string-to-format, args-to-format...)

#define STR(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)
#define CSTR(S) MAKE_CSTRING(CharT, S)

template <class T>
struct context {};

template <>
struct context<char> {
  using type = std::format_context;
};

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
template <>
struct context<wchar_t> {
  using type = std::wformat_context;
};
#endif

template <class T>
using context_t = typename context<T>::type;

// A user-defined type used to test the handle formatter.
enum class status : uint16_t { foo = 0xAAAA, bar = 0x5555, foobar = 0xAA55 };

// The formatter for a user-defined type used to test the handle formatter.
template <class CharT>
struct std::formatter<status, CharT> {
  int type = 0;

  constexpr auto parse(basic_format_parse_context<CharT>& parse_ctx) -> decltype(parse_ctx.begin()) {
    auto begin = parse_ctx.begin();
    auto end = parse_ctx.end();
    if (begin == end)
      return begin;

    switch (*begin) {
    case CharT('x'):
      break;
    case CharT('X'):
      type = 1;
      break;
    case CharT('s'):
      type = 2;
      break;
    case CharT('}'):
      return begin;
    default:
      throw_format_error("The format-spec type has a type not supported for a status argument");
    }

    ++begin;
    if (begin != end && *begin != CharT('}'))
      throw_format_error("The format-spec should consume the input or end with a '}'");

    return begin;
  }

  template <class Out>
  auto format(status s, basic_format_context<Out, CharT>& ctx) -> decltype(ctx.out()) {
    const char* names[] = {"foo", "bar", "foobar"};
    char buffer[7];
    const char* begin = names[0];
    const char* end = names[0];
    switch (type) {
    case 0:
      begin = buffer;
      buffer[0] = '0';
      buffer[1] = 'x';
      end = std::to_chars(&buffer[2], std::end(buffer), static_cast<uint16_t>(s), 16).ptr;
      buffer[6] = '\0';
      break;

    case 1:
      begin = buffer;
      buffer[0] = '0';
      buffer[1] = 'X';
      end = std::to_chars(&buffer[2], std::end(buffer), static_cast<uint16_t>(s), 16).ptr;
      std::transform(static_cast<const char*>(&buffer[2]), end, &buffer[2], [](char c) {
        return static_cast<char>(std::toupper(c)); });
      buffer[6] = '\0';
      break;

    case 2:
      switch (s) {
      case status::foo:
        begin = names[0];
        break;
      case status::bar:
        begin = names[1];
        break;
      case status::foobar:
        begin = names[2];
        break;
      }
      end = begin + strlen(begin);
      break;
    }

    return std::copy(begin, end, ctx.out());
  }

private:
  void throw_format_error(const char* s) {
#ifndef TEST_HAS_NO_EXCEPTIONS
    throw std::format_error(s);
#else
    (void)s;
    std::abort();
#endif
  }
};

template <class CharT>
std::vector<std::basic_string_view<CharT>> invalid_types(std::string valid) {
  std::vector<std::basic_string_view<CharT>> result;

#define CASE(T)                                                                                                        \
case #T[0]:                                                                                                            \
  result.push_back(SV("Invalid formatter type {:" #T "}"));                                                            \
  break;

  for (auto type : "aAbBcdeEfFgGopsxX") {
    if (valid.find(type) != std::string::npos)
      continue;

    switch (type) {
      CASE(a)
      CASE(A)
      CASE(b)
      CASE(B)
      CASE(c)
      CASE(d)
      CASE(e)
      CASE(E)
      CASE(f)
      CASE(F)
      CASE(g)
      CASE(G)
      CASE(o)
      CASE(p)
      CASE(s)
      CASE(x)
      CASE(X)
    case 0:
      break;
    default:
      assert(false && "Add the type to the list of cases.");
    }
  }
#undef CASE

  return result;
}

// Using a const ref for world and universe so a string literal will be a character array.
// When passed as character array W and U have different types.
template <class CharT, class W, class U, class TestFunction, class ExceptionTest>
void format_test_string(const W& world, const U& universe, TestFunction check, ExceptionTest check_exception) {

  // *** Valid input tests ***
  // Unsed argument is ignored. TODO FMT what does the Standard mandate?
  check.template operator()<"hello {}">(SV("hello world"), world, universe);
  check.template operator()<"hello {} and {}">(SV("hello world and universe"), world, universe);
  check.template operator()<"hello {0}">(SV("hello world"), world, universe);
  check.template operator()<"hello {1}">(SV("hello universe"), world, universe);
  check.template operator()<"hello {1} and {0}">(SV("hello universe and world"), world, universe);

  check.template operator()<"hello {:_>}">(SV("hello world"), world);
  check.template operator()<"hello {:>8}">(SV("hello    world"), world);
  check.template operator()<"hello {:_>8}">(SV("hello ___world"), world);
  check.template operator()<"hello {:_^8}">(SV("hello _world__"), world);
  check.template operator()<"hello {:_<8}">(SV("hello world___"), world);

  check.template operator()<"hello {:>>8}">(SV("hello >>>world"), world);
  check.template operator()<"hello {:<>8}">(SV("hello <<<world"), world);
  check.template operator()<"hello {:^>8}">(SV("hello ^^^world"), world);

  check.template operator()<"hello {:$>{}}">(SV("hello $world"), world, 6);
  check.template operator()<"hello {0:$>{1}}">(SV("hello $world"), world, 6);
  check.template operator()<"hello {1:$>{0}}">(SV("hello $world"), 6, world);

  check.template operator()<"hello {:.5}">(SV("hello world"), world);
  check.template operator()<"hello {:.5}">(SV("hello unive"), universe);

  check.template operator()<"hello {:.{}}">(SV("hello univer"), universe, 6);
  check.template operator()<"hello {0:.{1}}">(SV("hello univer"), universe, 6);
  check.template operator()<"hello {1:.{0}}">(SV("hello univer"), 6, universe);

  check.template operator()<"hello {:%^7.7}">(SV("hello %world%"), world);
  check.template operator()<"hello {:%^7.7}">(SV("hello univers"), universe);
  check.template operator()<"hello {:%^{}.{}}">(SV("hello %world%"), world, 7, 7);
  check.template operator()<"hello {0:%^{1}.{2}}">(SV("hello %world%"), world, 7, 7);
  check.template operator()<"hello {0:%^{2}.{1}}">(SV("hello %world%"), world, 7, 7);
  check.template operator()<"hello {1:%^{0}.{2}}">(SV("hello %world%"), 7, world, 7);

  check.template operator()<"hello {:_>s}">(SV("hello world"), world);
  check.template operator()<"hello {:$>{}s}">(SV("hello $world"), world, 6);
  check.template operator()<"hello {:.5s}">(SV("hello world"), world);
  check.template operator()<"hello {:.{}s}">(SV("hello univer"), universe, 6);
  check.template operator()<"hello {:%^7.7s}">(SV("hello %world%"), world);

  check.template operator()<"hello {:#>8.3s}">(SV("hello #####uni"), universe);
  check.template operator()<"hello {:#^8.3s}">(SV("hello ##uni###"), universe);
  check.template operator()<"hello {:#<8.3s}">(SV("hello uni#####"), universe);

  // *** sign ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("hello {:-}"), world);

  // *** alternate form ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("hello {:#}"), world);

  // *** zero-padding ***
  check_exception("A format-spec width field shouldn't have a leading zero", SV("hello {:0}"), world);

  // *** width ***
  // Width 0 allowed, but not useful for string arguments.
  check.template operator()<"hello {:{}}">(SV("hello world"), world, 0);

#ifdef _LIBCPP_VERSION
  // This limit isn't specified in the Standard.
  static_assert(std::__format::__number_max == 2'147'483'647, "Update the assert and the test.");
  check_exception("The numeric value of the format-spec is too large", SV("{:2147483648}"), world);
  check_exception("The numeric value of the format-spec is too large", SV("{:5000000000}"), world);
  check_exception("The numeric value of the format-spec is too large", SV("{:10000000000}"), world);
#endif

  check_exception("A format-spec arg-id replacement shouldn't have a negative value", SV("hello {:{}}"), world, -1);
  check_exception("A format-spec arg-id replacement exceeds the maximum supported value", SV("hello {:{}}"), world,
                  unsigned(-1));
  check_exception("Argument index out of bounds", SV("hello {:{}}"), world);
  check_exception("A format-spec arg-id replacement argument isn't an integral type", SV("hello {:{}}"), world,
                  universe);
  check_exception("Using manual argument numbering in automatic argument numbering mode", SV("hello {:{0}}"), world, 1);
  check_exception("Using automatic argument numbering in manual argument numbering mode", SV("hello {0:{}}"), world, 1);
  // Arg-id may not have leading zeros.
  check_exception("Invalid arg-id", SV("hello {0:{01}}"), world, 1);

  // *** precision ***
#ifdef _LIBCPP_VERSION
  // This limit isn't specified in the Standard.
  static_assert(std::__format::__number_max == 2'147'483'647, "Update the assert and the test.");
  check_exception("The numeric value of the format-spec is too large", SV("{:.2147483648}"), world);
  check_exception("The numeric value of the format-spec is too large", SV("{:.5000000000}"), world);
  check_exception("The numeric value of the format-spec is too large", SV("{:.10000000000}"), world);
#endif

  // Precision 0 allowed, but not useful for string arguments.
  check.template operator()<"hello {:.{}}">(SV("hello "), world, 0);
  // Precision may have leading zeros. Secondly tests the value is still base 10.
  check.template operator()<"hello {:.000010}">(SV("hello 0123456789"), STR("0123456789abcdef"));
  check_exception("A format-spec arg-id replacement shouldn't have a negative value", SV("hello {:.{}}"), world, -1);
  check_exception("A format-spec arg-id replacement exceeds the maximum supported value", SV("hello {:.{}}"), world,
                  ~0u);
  check_exception("Argument index out of bounds", SV("hello {:.{}}"), world);
  check_exception("A format-spec arg-id replacement argument isn't an integral type", SV("hello {:.{}}"), world,
                  universe);
  check_exception("Using manual argument numbering in automatic argument numbering mode", SV("hello {:.{0}}"), world,
                  1);
  check_exception("Using automatic argument numbering in manual argument numbering mode", SV("hello {0:.{}}"), world,
                  1);
  // Arg-id may not have leading zeros.
  check_exception("Invalid arg-id", SV("hello {0:.{01}}"), world, 1);

  // *** locale-specific form ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("hello {:L}"), world);

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("s"))
    check_exception("The format-spec type has a type not supported for a string argument", fmt, world);
}

template <class CharT, class TestFunction>
void format_test_string_unicode([[maybe_unused]] TestFunction check) {
  // unicode.pass.cpp and ascii.pass.cpp have additional tests.
#ifndef TEST_HAS_NO_UNICODE
  // Make sure all possible types are tested. For clarity don't use macros.
  if constexpr (std::same_as<CharT, char>) {
    const char* c_string = "aßc";
    check.template operator()<"{:*^5}">(SV("*aßc*"), c_string);
    check.template operator()<"{:*^4.2}">(SV("*aß*"), c_string);

    check.template operator()<"{:*^5}">(SV("*aßc*"), const_cast<char*>(c_string));
    check.template operator()<"{:*^4.2}">(SV("*aß*"), const_cast<char*>(c_string));

    check.template operator()<"{:*^5}">(SV("*aßc*"), "aßc");
    check.template operator()<"{:*^4.2}">(SV("*aß*"), "aßc");

    check.template operator()<"{:*^5}">(SV("*aßc*"), std::string("aßc"));
    check.template operator()<"{:*^4.2}">(SV("*aß*"), std::string("aßc"));

    check.template operator()<"{:*^5}">(SV("*aßc*"), std::string_view("aßc"));
    check.template operator()<"{:*^4.2}">(SV("*aß*"), std::string_view("aßc"));
  }
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
  else {
    const wchar_t* c_string = L"aßc";
    check.template operator()<"{:*^5}">(SV("*aßc*"), c_string);
    check.template operator()<"{:*^4.2}">(SV("*aß*"), c_string);

    check.template operator()<"{:*^5}">(SV("*aßc*"), const_cast<wchar_t*>(c_string));
    check.template operator()<"{:*^4.2}">(SV("*aß*"), const_cast<wchar_t*>(c_string));

    check.template operator()<"{:*^5}">(SV("*aßc*"), L"aßc");
    check.template operator()<"{:*^4.2}">(SV("*aß*"), L"aßc");

    check.template operator()<"{:*^5}">(SV("*aßc*"), std::wstring(L"aßc"));
    check.template operator()<"{:*^4.2}">(SV("*aß*"), std::wstring(L"aßc"));

    check.template operator()<"{:*^5}">(SV("*aßc*"), std::wstring_view(L"aßc"));
    check.template operator()<"{:*^4.2}">(SV("*aß*"), std::wstring_view(L"aßc"));
  }
#  endif // TEST_HAS_NO_WIDE_CHARACTERS

  // ß requires one column
  check.template operator()<"{}">(SV("aßc"), STR("aßc"));

  check.template operator()<"{:.3}">(SV("aßc"), STR("aßc"));
  check.template operator()<"{:.2}">(SV("aß"), STR("aßc"));
  check.template operator()<"{:.1}">(SV("a"), STR("aßc"));

  check.template operator()<"{:3.3}">(SV("aßc"), STR("aßc"));
  check.template operator()<"{:2.2}">(SV("aß"), STR("aßc"));
  check.template operator()<"{:1.1}">(SV("a"), STR("aßc"));

  check.template operator()<"{:-<6}">(SV("aßc---"), STR("aßc"));
  check.template operator()<"{:-^6}">(SV("-aßc--"), STR("aßc"));
  check.template operator()<"{:->6}">(SV("---aßc"), STR("aßc"));

  // \u1000 requires two columns
  check.template operator()<"{}">(SV("a\u1110c"), STR("a\u1110c"));

  check.template operator()<"{:.4}">(SV("a\u1100c"), STR("a\u1100c"));
  check.template operator()<"{:.3}">(SV("a\u1100"), STR("a\u1100c"));
  check.template operator()<"{:.2}">(SV("a"), STR("a\u1100c"));
  check.template operator()<"{:.1}">(SV("a"), STR("a\u1100c"));

  check.template operator()<"{:-<4.4}">(SV("a\u1100c"), STR("a\u1100c"));
  check.template operator()<"{:-<3.3}">(SV("a\u1100"), STR("a\u1100c"));
  check.template operator()<"{:-<2.2}">(SV("a-"), STR("a\u1100c"));
  check.template operator()<"{:-<1.1}">(SV("a"), STR("a\u1100c"));

  check.template operator()<"{:-<7}">(SV("a\u1110c---"), STR("a\u1110c"));
  check.template operator()<"{:-^7}">(SV("-a\u1110c--"), STR("a\u1110c"));
  check.template operator()<"{:->7}">(SV("---a\u1110c"), STR("a\u1110c"));

  // Examples used in P1868R2
  check.template operator()<"{:*^3}">(SV("*\u0041*"), STR("\u0041")); // { LATIN CAPITAL LETTER A }
  check.template operator()<"{:*^3}">(SV("*\u00c1*"), STR("\u00c1")); // { LATIN CAPITAL LETTER A WITH ACUTE }
  check.template operator()<"{:*^3}">(
      SV("*\u0041\u0301*"),
      STR("\u0041\u0301")); // { LATIN CAPITAL LETTER A } { COMBINING ACUTE ACCENT }
  check.template operator()<"{:*^3}">(SV("*\u0132*"), STR("\u0132")); // { LATIN CAPITAL LIGATURE IJ }
  check.template operator()<"{:*^3}">(SV("*\u0394*"), STR("\u0394")); // { GREEK CAPITAL LETTER DELTA }

  check.template operator()<"{:*^3}">(SV("*\u0429*"), STR("\u0429"));         // { CYRILLIC CAPITAL LETTER SHCHA }
  check.template operator()<"{:*^3}">(SV("*\u05d0*"), STR("\u05d0"));         // { HEBREW LETTER ALEF }
  check.template operator()<"{:*^3}">(SV("*\u0634*"), STR("\u0634"));         // { ARABIC LETTER SHEEN }
  check.template operator()<"{:*^4}">(SV("*\u3009*"), STR("\u3009"));         // { RIGHT-POINTING ANGLE BRACKET }
  check.template operator()<"{:*^4}">(SV("*\u754c*"), STR("\u754c"));         // { CJK Unified Ideograph-754C }
  check.template operator()<"{:*^4}">(SV("*\U0001f921*"), STR("\U0001f921")); // { UNICORN FACE }
  check.template operator()<"{:*^4}">(
      SV("*\U0001f468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466*"),
      STR("\U0001f468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466")); // { Family: Man, Woman, Girl, Boy }
#endif // TEST_HAS_NO_UNICODE
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_string_tests(TestFunction check, ExceptionTest check_exception) {
  std::basic_string<CharT> world = STR("world");
  std::basic_string<CharT> universe = STR("universe");

  // Test a string literal in a way it won't decay to a pointer.
  if constexpr (std::same_as<CharT, char>)
    format_test_string<CharT>("world", "universe", check, check_exception);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  else
    format_test_string<CharT>(L"world", L"universe", check, check_exception);
#endif

  format_test_string<CharT>(world.c_str(), universe.c_str(), check, check_exception);
  format_test_string<CharT>(const_cast<CharT*>(world.c_str()), const_cast<CharT*>(universe.c_str()), check,
                            check_exception);
  format_test_string<CharT>(std::basic_string_view<CharT>(world), std::basic_string_view<CharT>(universe), check,
                            check_exception);
  format_test_string<CharT>(world, universe, check, check_exception);
  format_test_string_unicode<CharT>(check);
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_bool(TestFunction check, ExceptionTest check_exception) {

  // *** align-fill & width ***
  check.template operator()<"answer is '{:7}'">(SV("answer is 'true   '"), true);
  check.template operator()<"answer is '{:>7}'">(SV("answer is '   true'"), true);
  check.template operator()<"answer is '{:<7}'">(SV("answer is 'true   '"), true);
  check.template operator()<"answer is '{:^7}'">(SV("answer is ' true  '"), true);

  check.template operator()<"answer is '{:8s}'">(SV("answer is 'false   '"), false);
  check.template operator()<"answer is '{:>8s}'">(SV("answer is '   false'"), false);
  check.template operator()<"answer is '{:<8s}'">(SV("answer is 'false   '"), false);
  check.template operator()<"answer is '{:^8s}'">(SV("answer is ' false  '"), false);

  check.template operator()<"answer is '{:->7}'">(SV("answer is '---true'"), true);
  check.template operator()<"answer is '{:-<7}'">(SV("answer is 'true---'"), true);
  check.template operator()<"answer is '{:-^7}'">(SV("answer is '-true--'"), true);

  check.template operator()<"answer is '{:->8s}'">(SV("answer is '---false'"), false);
  check.template operator()<"answer is '{:-<8s}'">(SV("answer is 'false---'"), false);
  check.template operator()<"answer is '{:-^8s}'">(SV("answer is '-false--'"), false);

  // *** Sign ***
  check_exception("A sign field isn't allowed in this format-spec", SV("{:-}"), true);
  check_exception("A sign field isn't allowed in this format-spec", SV("{:+}"), true);
  check_exception("A sign field isn't allowed in this format-spec", SV("{: }"), true);

  check_exception("A sign field isn't allowed in this format-spec", SV("{:-s}"), true);
  check_exception("A sign field isn't allowed in this format-spec", SV("{:+s}"), true);
  check_exception("A sign field isn't allowed in this format-spec", SV("{: s}"), true);

  // *** alternate form ***
  check_exception("An alternate form field isn't allowed in this format-spec", SV("{:#}"), true);
  check_exception("An alternate form field isn't allowed in this format-spec", SV("{:#s}"), true);

  // *** zero-padding ***
  check_exception("A zero-padding field isn't allowed in this format-spec", SV("{:0}"), true);
  check_exception("A zero-padding field isn't allowed in this format-spec", SV("{:0s}"), true);

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.0}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.42}"), true);

  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.s}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.0s}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.42s}"), true);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBdosxX"))
    check_exception("The format-spec type has a type not supported for a bool argument", fmt, true);
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_bool_as_integer(TestFunction check, ExceptionTest check_exception) {
  // *** align-fill & width ***
  check.template operator()<"answer is '{:<1d}'">(SV("answer is '1'"), true);
  check.template operator()<"answer is '{:<2d}'">(SV("answer is '1 '"), true);
  check.template operator()<"answer is '{:<2d}'">(SV("answer is '0 '"), false);

  check.template operator()<"answer is '{:6d}'">(SV("answer is '     1'"), true);
  check.template operator()<"answer is '{:>6d}'">(SV("answer is '     1'"), true);
  check.template operator()<"answer is '{:<6d}'">(SV("answer is '1     '"), true);
  check.template operator()<"answer is '{:^6d}'">(SV("answer is '  1   '"), true);

  check.template operator()<"answer is '{:*>6d}'">(SV("answer is '*****0'"), false);
  check.template operator()<"answer is '{:*<6d}'">(SV("answer is '0*****'"), false);
  check.template operator()<"answer is '{:*^6d}'">(SV("answer is '**0***'"), false);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>06d}'">(SV("answer is '     1'"), true);
  check.template operator()<"answer is '{:<06d}'">(SV("answer is '1     '"), true);
  check.template operator()<"answer is '{:^06d}'">(SV("answer is '  1   '"), true);

  // *** Sign ***
  check.template operator()<"answer is {:d}">(SV("answer is 1"), true);
  check.template operator()<"answer is {:-d}">(SV("answer is 0"), false);
  check.template operator()<"answer is {:+d}">(SV("answer is +1"), true);
  check.template operator()<"answer is {: d}">(SV("answer is  0"), false);

  // *** alternate form ***
  check.template operator()<"answer is {:+#d}">(SV("answer is +1"), true);
  check.template operator()<"answer is {:+b}">(SV("answer is +1"), true);
  check.template operator()<"answer is {:+#b}">(SV("answer is +0b1"), true);
  check.template operator()<"answer is {:+#B}">(SV("answer is +0B1"), true);
  check.template operator()<"answer is {:+o}">(SV("answer is +1"), true);
  check.template operator()<"answer is {:+#o}">(SV("answer is +01"), true);
  check.template operator()<"answer is {:+x}">(SV("answer is +1"), true);
  check.template operator()<"answer is {:+#x}">(SV("answer is +0x1"), true);
  check.template operator()<"answer is {:+X}">(SV("answer is +1"), true);
  check.template operator()<"answer is {:+#X}">(SV("answer is +0X1"), true);

  check.template operator()<"answer is {:#d}">(SV("answer is 0"), false);
  check.template operator()<"answer is {:b}">(SV("answer is 0"), false);
  check.template operator()<"answer is {:#b}">(SV("answer is 0b0"), false);
  check.template operator()<"answer is {:#B}">(SV("answer is 0B0"), false);
  check.template operator()<"answer is {:o}">(SV("answer is 0"), false);
  check.template operator()<"answer is {:#o}">(SV("answer is 0"), false);
  check.template operator()<"answer is {:x}">(SV("answer is 0"), false);
  check.template operator()<"answer is {:#x}">(SV("answer is 0x0"), false);
  check.template operator()<"answer is {:X}">(SV("answer is 0"), false);
  check.template operator()<"answer is {:#X}">(SV("answer is 0X0"), false);

  // *** zero-padding & width ***
  check.template operator()<"answer is {:+#012d}">(SV("answer is +00000000001"), true);
  check.template operator()<"answer is {:+012b}">(SV("answer is +00000000001"), true);
  check.template operator()<"answer is {:+#012b}">(SV("answer is +0b000000001"), true);
  check.template operator()<"answer is {:+#012B}">(SV("answer is +0B000000001"), true);
  check.template operator()<"answer is {:+012o}">(SV("answer is +00000000001"), true);
  check.template operator()<"answer is {:+#012o}">(SV("answer is +00000000001"), true);
  check.template operator()<"answer is {:+012x}">(SV("answer is +00000000001"), true);
  check.template operator()<"answer is {:+#012x}">(SV("answer is +0x000000001"), true);
  check.template operator()<"answer is {:+012X}">(SV("answer is +00000000001"), true);
  check.template operator()<"answer is {:+#012X}">(SV("answer is +0X000000001"), true);

  check.template operator()<"answer is {:#012d}">(SV("answer is 000000000000"), false);
  check.template operator()<"answer is {:012b}">(SV("answer is 000000000000"), false);
  check.template operator()<"answer is {:#012b}">(SV("answer is 0b0000000000"), false);
  check.template operator()<"answer is {:#012B}">(SV("answer is 0B0000000000"), false);
  check.template operator()<"answer is {:012o}">(SV("answer is 000000000000"), false);
  check.template operator()<"answer is {:#012o}">(SV("answer is 000000000000"), false);
  check.template operator()<"answer is {:012x}">(SV("answer is 000000000000"), false);
  check.template operator()<"answer is {:#012x}">(SV("answer is 0x0000000000"), false);
  check.template operator()<"answer is {:012X}">(SV("answer is 000000000000"), false);
  check.template operator()<"answer is {:#012X}">(SV("answer is 0X0000000000"), false);

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.0}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.42}"), true);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdosxX"))
    check_exception("The format-spec type has a type not supported for a bool argument", fmt, true);
}

template <class I, class CharT, class TestFunction, class ExceptionTest>
void format_test_integer_as_integer(TestFunction check, ExceptionTest check_exception) {
  // *** align-fill & width ***
  check.template operator()<"answer is '{:<1}'">(SV("answer is '42'"), I(42));
  check.template operator()<"answer is '{:<2}'">(SV("answer is '42'"), I(42));
  check.template operator()<"answer is '{:<3}'">(SV("answer is '42 '"), I(42));

  check.template operator()<"answer is '{:7}'">(SV("answer is '     42'"), I(42));
  check.template operator()<"answer is '{:>7}'">(SV("answer is '     42'"), I(42));
  check.template operator()<"answer is '{:<7}'">(SV("answer is '42     '"), I(42));
  check.template operator()<"answer is '{:^7}'">(SV("answer is '  42   '"), I(42));

  check.template operator()<"answer is '{:*>7}'">(SV("answer is '*****42'"), I(42));
  check.template operator()<"answer is '{:*<7}'">(SV("answer is '42*****'"), I(42));
  check.template operator()<"answer is '{:*^7}'">(SV("answer is '**42***'"), I(42));

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>07}'">(SV("answer is '     42'"), I(42));
  check.template operator()<"answer is '{:<07}'">(SV("answer is '42     '"), I(42));
  check.template operator()<"answer is '{:^07}'">(SV("answer is '  42   '"), I(42));

  // *** Sign ***
  if constexpr (std::signed_integral<I>)
    check.template operator()<"answer is {}">(SV("answer is -42"), I(-42));
  check.template operator()<"answer is {}">(SV("answer is 0"), I(0));
  check.template operator()<"answer is {}">(SV("answer is 42"), I(42));

  if constexpr (std::signed_integral<I>)
    check.template operator()<"answer is {:-}">(SV("answer is -42"), I(-42));
  check.template operator()<"answer is {:-}">(SV("answer is 0"), I(0));
  check.template operator()<"answer is {:-}">(SV("answer is 42"), I(42));

  if constexpr (std::signed_integral<I>)
    check.template operator()<"answer is {:+}">(SV("answer is -42"), I(-42));
  check.template operator()<"answer is {:+}">(SV("answer is +0"), I(0));
  check.template operator()<"answer is {:+}">(SV("answer is +42"), I(42));

  if constexpr (std::signed_integral<I>)
    check.template operator()<"answer is {: }">(SV("answer is -42"), I(-42));
  check.template operator()<"answer is {: }">(SV("answer is  0"), I(0));
  check.template operator()<"answer is {: }">(SV("answer is  42"), I(42));

  // *** alternate form ***
  if constexpr (std::signed_integral<I>) {
    check.template operator()<"answer is {:#}">(SV("answer is -42"), I(-42));
    check.template operator()<"answer is {:#d}">(SV("answer is -42"), I(-42));
    check.template operator()<"answer is {:b}">(SV("answer is -101010"), I(-42));
    check.template operator()<"answer is {:#b}">(SV("answer is -0b101010"), I(-42));
    check.template operator()<"answer is {:#B}">(SV("answer is -0B101010"), I(-42));
    check.template operator()<"answer is {:o}">(SV("answer is -52"), I(-42));
    check.template operator()<"answer is {:#o}">(SV("answer is -052"), I(-42));
    check.template operator()<"answer is {:x}">(SV("answer is -2a"), I(-42));
    check.template operator()<"answer is {:#x}">(SV("answer is -0x2a"), I(-42));
    check.template operator()<"answer is {:X}">(SV("answer is -2A"), I(-42));
    check.template operator()<"answer is {:#X}">(SV("answer is -0X2A"), I(-42));
  }
  check.template operator()<"answer is {:#}">(SV("answer is 0"), I(0));
  check.template operator()<"answer is {:#d}">(SV("answer is 0"), I(0));
  check.template operator()<"answer is {:b}">(SV("answer is 0"), I(0));
  check.template operator()<"answer is {:#b}">(SV("answer is 0b0"), I(0));
  check.template operator()<"answer is {:#B}">(SV("answer is 0B0"), I(0));
  check.template operator()<"answer is {:o}">(SV("answer is 0"), I(0));
  check.template operator()<"answer is {:#o}">(SV("answer is 0"), I(0));
  check.template operator()<"answer is {:x}">(SV("answer is 0"), I(0));
  check.template operator()<"answer is {:#x}">(SV("answer is 0x0"), I(0));
  check.template operator()<"answer is {:X}">(SV("answer is 0"), I(0));
  check.template operator()<"answer is {:#X}">(SV("answer is 0X0"), I(0));

  check.template operator()<"answer is {:+#}">(SV("answer is +42"), I(42));
  check.template operator()<"answer is {:+#d}">(SV("answer is +42"), I(42));
  check.template operator()<"answer is {:+b}">(SV("answer is +101010"), I(42));
  check.template operator()<"answer is {:+#b}">(SV("answer is +0b101010"), I(42));
  check.template operator()<"answer is {:+#B}">(SV("answer is +0B101010"), I(42));
  check.template operator()<"answer is {:+o}">(SV("answer is +52"), I(42));
  check.template operator()<"answer is {:+#o}">(SV("answer is +052"), I(42));
  check.template operator()<"answer is {:+x}">(SV("answer is +2a"), I(42));
  check.template operator()<"answer is {:+#x}">(SV("answer is +0x2a"), I(42));
  check.template operator()<"answer is {:+X}">(SV("answer is +2A"), I(42));
  check.template operator()<"answer is {:+#X}">(SV("answer is +0X2A"), I(42));

  // *** zero-padding & width ***
  if constexpr (std::signed_integral<I>) {
    check.template operator()<"answer is {:#012}">(SV("answer is -00000000042"), I(-42));
    check.template operator()<"answer is {:#012d}">(SV("answer is -00000000042"), I(-42));
    check.template operator()<"answer is {:012b}">(SV("answer is -00000101010"), I(-42));
    check.template operator()<"answer is {:#012b}">(SV("answer is -0b000101010"), I(-42));
    check.template operator()<"answer is {:#012B}">(SV("answer is -0B000101010"), I(-42));
    check.template operator()<"answer is {:012o}">(SV("answer is -00000000052"), I(-42));
    check.template operator()<"answer is {:#012o}">(SV("answer is -00000000052"), I(-42));
    check.template operator()<"answer is {:012x}">(SV("answer is -0000000002a"), I(-42));
    check.template operator()<"answer is {:#012x}">(SV("answer is -0x00000002a"), I(-42));
    check.template operator()<"answer is {:012X}">(SV("answer is -0000000002A"), I(-42));
    check.template operator()<"answer is {:#012X}">(SV("answer is -0X00000002A"), I(-42));
  }

  check.template operator()<"answer is {:#012}">(SV("answer is 000000000000"), I(0));
  check.template operator()<"answer is {:#012d}">(SV("answer is 000000000000"), I(0));
  check.template operator()<"answer is {:012b}">(SV("answer is 000000000000"), I(0));
  check.template operator()<"answer is {:#012b}">(SV("answer is 0b0000000000"), I(0));
  check.template operator()<"answer is {:#012B}">(SV("answer is 0B0000000000"), I(0));
  check.template operator()<"answer is {:012o}">(SV("answer is 000000000000"), I(0));
  check.template operator()<"answer is {:#012o}">(SV("answer is 000000000000"), I(0));
  check.template operator()<"answer is {:012x}">(SV("answer is 000000000000"), I(0));
  check.template operator()<"answer is {:#012x}">(SV("answer is 0x0000000000"), I(0));
  check.template operator()<"answer is {:012X}">(SV("answer is 000000000000"), I(0));
  check.template operator()<"answer is {:#012X}">(SV("answer is 0X0000000000"), I(0));

  check.template operator()<"answer is {:+#012}">(SV("answer is +00000000042"), I(42));
  check.template operator()<"answer is {:+#012d}">(SV("answer is +00000000042"), I(42));
  check.template operator()<"answer is {:+012b}">(SV("answer is +00000101010"), I(42));
  check.template operator()<"answer is {:+#012b}">(SV("answer is +0b000101010"), I(42));
  check.template operator()<"answer is {:+#012B}">(SV("answer is +0B000101010"), I(42));
  check.template operator()<"answer is {:+012o}">(SV("answer is +00000000052"), I(42));
  check.template operator()<"answer is {:+#012o}">(SV("answer is +00000000052"), I(42));
  check.template operator()<"answer is {:+012x}">(SV("answer is +0000000002a"), I(42));
  check.template operator()<"answer is {:+#012x}">(SV("answer is +0x00000002a"), I(42));
  check.template operator()<"answer is {:+012X}">(SV("answer is +0000000002A"), I(42));
  check.template operator()<"answer is {:+#012X}">(SV("answer is +0X00000002A"), I(42));

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.}"), I(0));
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.0}"), I(0));
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.42}"), I(0));

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
    check_exception("The format-spec type has a type not supported for an integer argument", fmt, 42);
}

template <class I, class CharT, class TestFunction, class ExceptionTest>
void format_test_integer_as_char(TestFunction check, ExceptionTest check_exception) {
  // *** align-fill & width ***
  check.template operator()<"answer is '{:6c}'">(SV("answer is '*     '"), I(42));
  check.template operator()<"answer is '{:>6c}'">(SV("answer is '     *'"), I(42));
  check.template operator()<"answer is '{:<6c}'">(SV("answer is '*     '"), I(42));
  check.template operator()<"answer is '{:^6c}'">(SV("answer is '  *   '"), I(42));

  check.template operator()<"answer is '{:->6c}'">(SV("answer is '-----*'"), I(42));
  check.template operator()<"answer is '{:-<6c}'">(SV("answer is '*-----'"), I(42));
  check.template operator()<"answer is '{:-^6c}'">(SV("answer is '--*---'"), I(42));

  // *** Sign ***
  check.template operator()<"answer is {:c}">(SV("answer is *"), I(42));
  check_exception("A sign field isn't allowed in this format-spec", SV("answer is {:-c}"), I(42));
  check_exception("A sign field isn't allowed in this format-spec", SV("answer is {:+c}"), I(42));
  check_exception("A sign field isn't allowed in this format-spec", SV("answer is {: c}"), I(42));

  // *** alternate form ***
  check_exception("An alternate form field isn't allowed in this format-spec", SV("answer is {:#c}"), I(42));

  // *** zero-padding & width ***
  check_exception("A zero-padding field isn't allowed in this format-spec", SV("answer is {:01c}"), I(42));

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.c}"), I(0));
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.0c}"), I(0));
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.42c}"), I(0));

  // *** locale-specific form ***
  // Note it has no effect but it's allowed.
  check.template operator()<"answer is '{:Lc}'">(SV("answer is '*'"), I(42));

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
    check_exception("The format-spec type has a type not supported for an integer argument", fmt, I(42));

  // *** Validate range ***
  // The code has some duplications to keep the if statement readable.
  if constexpr (std::signed_integral<CharT>) {
    if constexpr (std::signed_integral<I> && sizeof(I) > sizeof(CharT)) {
      check_exception("Integral value outside the range of the char type", SV("{:c}"), std::numeric_limits<I>::min());
      check_exception("Integral value outside the range of the char type", SV("{:c}"), std::numeric_limits<I>::max());
    } else if constexpr (std::unsigned_integral<I> && sizeof(I) >= sizeof(CharT)) {
      check_exception("Integral value outside the range of the char type", SV("{:c}"), std::numeric_limits<I>::max());
    }
  } else if constexpr (sizeof(I) > sizeof(CharT)) {
    check_exception("Integral value outside the range of the char type", SV("{:c}"), std::numeric_limits<I>::max());
  }
}

template <class I, class CharT, class TestFunction, class ExceptionTest>
void format_test_integer(TestFunction check, ExceptionTest check_exception) {
  format_test_integer_as_integer<I, CharT>(check, check_exception);
  format_test_integer_as_char<I, CharT>(check, check_exception);
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_signed_integer(TestFunction check, ExceptionTest check_exception) {
  format_test_integer<signed char, CharT>(check, check_exception);
  format_test_integer<short, CharT>(check, check_exception);
  format_test_integer<int, CharT>(check, check_exception);
  format_test_integer<long, CharT>(check, check_exception);
  format_test_integer<long long, CharT>(check, check_exception);
#ifndef TEST_HAS_NO_INT128
  format_test_integer<__int128_t, CharT>(check, check_exception);
#endif
  // *** check the minma and maxima ***
  check.template operator()<"{:#b}">(SV("-0b10000000"), std::numeric_limits<int8_t>::min());
  check.template operator()<"{:#o}">(SV("-0200"), std::numeric_limits<int8_t>::min());
  check.template operator()<"{:#}">(SV("-128"), std::numeric_limits<int8_t>::min());
  check.template operator()<"{:#x}">(SV("-0x80"), std::numeric_limits<int8_t>::min());

  check.template operator()<"{:#b}">(SV("-0b1000000000000000"), std::numeric_limits<int16_t>::min());
  check.template operator()<"{:#o}">(SV("-0100000"), std::numeric_limits<int16_t>::min());
  check.template operator()<"{:#}">(SV("-32768"), std::numeric_limits<int16_t>::min());
  check.template operator()<"{:#x}">(SV("-0x8000"), std::numeric_limits<int16_t>::min());

  check.template operator()<"{:#b}">(SV("-0b10000000000000000000000000000000"), std::numeric_limits<int32_t>::min());
  check.template operator()<"{:#o}">(SV("-020000000000"), std::numeric_limits<int32_t>::min());
  check.template operator()<"{:#}">(SV("-2147483648"), std::numeric_limits<int32_t>::min());
  check.template operator()<"{:#x}">(SV("-0x80000000"), std::numeric_limits<int32_t>::min());

  check.template operator()<"{:#b}">(SV("-0b1000000000000000000000000000000000000000000000000000000000000000"),
                                     std::numeric_limits<int64_t>::min());
  check.template operator()<"{:#o}">(SV("-01000000000000000000000"), std::numeric_limits<int64_t>::min());
  check.template operator()<"{:#}">(SV("-9223372036854775808"), std::numeric_limits<int64_t>::min());
  check.template operator()<"{:#x}">(SV("-0x8000000000000000"), std::numeric_limits<int64_t>::min());

#ifndef TEST_HAS_NO_INT128
  check.template operator()<"{:#b}">(
      SV("-0b1000000000000000000000000000000000000000000000000000000000000000"
         "0000000000000000000000000000000000000000000000000000000000000000"),
      std::numeric_limits<__int128_t>::min());
  check.template
  operator()<"{:#o}">(SV("-02000000000000000000000000000000000000000000"), std::numeric_limits<__int128_t>::min());
  check.template
  operator()<"{:#}">(SV("-170141183460469231731687303715884105728"), std::numeric_limits<__int128_t>::min());
  check.template operator()<"{:#x}">(SV("-0x80000000000000000000000000000000"), std::numeric_limits<__int128_t>::min());
#endif

  check.template operator()<"{:#b}">(SV("0b1111111"), std::numeric_limits<int8_t>::max());
  check.template operator()<"{:#o}">(SV("0177"), std::numeric_limits<int8_t>::max());
  check.template operator()<"{:#}">(SV("127"), std::numeric_limits<int8_t>::max());
  check.template operator()<"{:#x}">(SV("0x7f"), std::numeric_limits<int8_t>::max());

  check.template operator()<"{:#b}">(SV("0b111111111111111"), std::numeric_limits<int16_t>::max());
  check.template operator()<"{:#o}">(SV("077777"), std::numeric_limits<int16_t>::max());
  check.template operator()<"{:#}">(SV("32767"), std::numeric_limits<int16_t>::max());
  check.template operator()<"{:#x}">(SV("0x7fff"), std::numeric_limits<int16_t>::max());

  check.template operator()<"{:#b}">(SV("0b1111111111111111111111111111111"), std::numeric_limits<int32_t>::max());
  check.template operator()<"{:#o}">(SV("017777777777"), std::numeric_limits<int32_t>::max());
  check.template operator()<"{:#}">(SV("2147483647"), std::numeric_limits<int32_t>::max());
  check.template operator()<"{:#x}">(SV("0x7fffffff"), std::numeric_limits<int32_t>::max());

  check.template operator()<"{:#b}">(SV("0b111111111111111111111111111111111111111111111111111111111111111"),
                                     std::numeric_limits<int64_t>::max());
  check.template operator()<"{:#o}">(SV("0777777777777777777777"), std::numeric_limits<int64_t>::max());
  check.template operator()<"{:#}">(SV("9223372036854775807"), std::numeric_limits<int64_t>::max());
  check.template operator()<"{:#x}">(SV("0x7fffffffffffffff"), std::numeric_limits<int64_t>::max());

#ifndef TEST_HAS_NO_INT128
  check.template operator()<"{:#b}">(
      SV("0b111111111111111111111111111111111111111111111111111111111111111"
         "1111111111111111111111111111111111111111111111111111111111111111"),
      std::numeric_limits<__int128_t>::max());
  check.template
  operator()<"{:#o}">(SV("01777777777777777777777777777777777777777777"), std::numeric_limits<__int128_t>::max());
  check.template
  operator()<"{:#}">(SV("170141183460469231731687303715884105727"), std::numeric_limits<__int128_t>::max());
  check.template operator()<"{:#x}">(SV("0x7fffffffffffffffffffffffffffffff"), std::numeric_limits<__int128_t>::max());
#endif
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_unsigned_integer(TestFunction check, ExceptionTest check_exception) {
  format_test_integer<unsigned char, CharT>(check, check_exception);
  format_test_integer<unsigned short, CharT>(check, check_exception);
  format_test_integer<unsigned, CharT>(check, check_exception);
  format_test_integer<unsigned long, CharT>(check, check_exception);
  format_test_integer<unsigned long long, CharT>(check, check_exception);
#ifndef TEST_HAS_NO_INT128
  format_test_integer<__uint128_t, CharT>(check, check_exception);
#endif
  // *** test the maxima ***
  check.template operator()<"{:#b}">(SV("0b11111111"), std::numeric_limits<uint8_t>::max());
  check.template operator()<"{:#o}">(SV("0377"), std::numeric_limits<uint8_t>::max());
  check.template operator()<"{:#}">(SV("255"), std::numeric_limits<uint8_t>::max());
  check.template operator()<"{:#x}">(SV("0xff"), std::numeric_limits<uint8_t>::max());

  check.template operator()<"{:#b}">(SV("0b1111111111111111"), std::numeric_limits<uint16_t>::max());
  check.template operator()<"{:#o}">(SV("0177777"), std::numeric_limits<uint16_t>::max());
  check.template operator()<"{:#}">(SV("65535"), std::numeric_limits<uint16_t>::max());
  check.template operator()<"{:#x}">(SV("0xffff"), std::numeric_limits<uint16_t>::max());

  check.template operator()<"{:#b}">(SV("0b11111111111111111111111111111111"), std::numeric_limits<uint32_t>::max());
  check.template operator()<"{:#o}">(SV("037777777777"), std::numeric_limits<uint32_t>::max());
  check.template operator()<"{:#}">(SV("4294967295"), std::numeric_limits<uint32_t>::max());
  check.template operator()<"{:#x}">(SV("0xffffffff"), std::numeric_limits<uint32_t>::max());

  check.template operator()<"{:#b}">(SV("0b1111111111111111111111111111111111111111111111111111111111111111"),
                                     std::numeric_limits<uint64_t>::max());
  check.template operator()<"{:#o}">(SV("01777777777777777777777"), std::numeric_limits<uint64_t>::max());
  check.template operator()<"{:#}">(SV("18446744073709551615"), std::numeric_limits<uint64_t>::max());
  check.template operator()<"{:#x}">(SV("0xffffffffffffffff"), std::numeric_limits<uint64_t>::max());

#ifndef TEST_HAS_NO_INT128
  check.template operator()<"{:#b}">(
      SV("0b1111111111111111111111111111111111111111111111111111111111111111"
         "1111111111111111111111111111111111111111111111111111111111111111"),
      std::numeric_limits<__uint128_t>::max());
  check.template
  operator()<"{:#o}">(SV("03777777777777777777777777777777777777777777"), std::numeric_limits<__uint128_t>::max());
  check.template
  operator()<"{:#}">(SV("340282366920938463463374607431768211455"), std::numeric_limits<__uint128_t>::max());
  check.template operator()<"{:#x}">(SV("0xffffffffffffffffffffffffffffffff"), std::numeric_limits<__uint128_t>::max());
#endif
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_char(TestFunction check, ExceptionTest check_exception) {

  // ***** Char type *****
  // *** align-fill & width ***
  check.template operator()<"answer is '{:6}'">(SV("answer is '*     '"), CharT('*'));

  check.template operator()<"answer is '{:>6}'">(SV("answer is '     *'"), CharT('*'));
  check.template operator()<"answer is '{:<6}'">(SV("answer is '*     '"), CharT('*'));
  check.template operator()<"answer is '{:^6}'">(SV("answer is '  *   '"), CharT('*'));

  check.template operator()<"answer is '{:6c}'">(SV("answer is '*     '"), CharT('*'));
  check.template operator()<"answer is '{:>6c}'">(SV("answer is '     *'"), CharT('*'));
  check.template operator()<"answer is '{:<6c}'">(SV("answer is '*     '"), CharT('*'));
  check.template operator()<"answer is '{:^6c}'">(SV("answer is '  *   '"), CharT('*'));

  check.template operator()<"answer is '{:->6}'">(SV("answer is '-----*'"), CharT('*'));
  check.template operator()<"answer is '{:-<6}'">(SV("answer is '*-----'"), CharT('*'));
  check.template operator()<"answer is '{:-^6}'">(SV("answer is '--*---'"), CharT('*'));

  check.template operator()<"answer is '{:->6c}'">(SV("answer is '-----*'"), CharT('*'));
  check.template operator()<"answer is '{:-<6c}'">(SV("answer is '*-----'"), CharT('*'));
  check.template operator()<"answer is '{:-^6c}'">(SV("answer is '--*---'"), CharT('*'));

  // *** Sign ***
  check_exception("A sign field isn't allowed in this format-spec", SV("{:-}"), CharT('*'));
  check_exception("A sign field isn't allowed in this format-spec", SV("{:+}"), CharT('*'));
  check_exception("A sign field isn't allowed in this format-spec", SV("{: }"), CharT('*'));

  check_exception("A sign field isn't allowed in this format-spec", SV("{:-c}"), CharT('*'));
  check_exception("A sign field isn't allowed in this format-spec", SV("{:+c}"), CharT('*'));
  check_exception("A sign field isn't allowed in this format-spec", SV("{: c}"), CharT('*'));

  // *** alternate form ***
  check_exception("An alternate form field isn't allowed in this format-spec", SV("{:#}"), CharT('*'));
  check_exception("An alternate form field isn't allowed in this format-spec", SV("{:#c}"), CharT('*'));

  // *** zero-padding ***
  check_exception("A zero-padding field isn't allowed in this format-spec", SV("{:0}"), CharT('*'));
  check_exception("A zero-padding field isn't allowed in this format-spec", SV("{:0c}"), CharT('*'));

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.0}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.42}"), CharT('*'));

  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.c}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.0c}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.42c}"), CharT('*'));

  // *** locale-specific form ***
  // Note it has no effect but it's allowed.
  check.template operator()<"answer is '{:L}'">(SV("answer is '*'"), '*');
  check.template operator()<"answer is '{:Lc}'">(SV("answer is '*'"), '*');

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
    check_exception("The format-spec type has a type not supported for a char argument", fmt, CharT('*'));
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_char_as_integer(TestFunction check, ExceptionTest check_exception) {
  // *** align-fill & width ***
  check.template operator()<"answer is '{:<1d}'">(SV("answer is '42'"), CharT('*'));

  check.template operator()<"answer is '{:<2d}'">(SV("answer is '42'"), CharT('*'));
  check.template operator()<"answer is '{:<3d}'">(SV("answer is '42 '"), CharT('*'));

  check.template operator()<"answer is '{:7d}'">(SV("answer is '     42'"), CharT('*'));
  check.template operator()<"answer is '{:>7d}'">(SV("answer is '     42'"), CharT('*'));
  check.template operator()<"answer is '{:<7d}'">(SV("answer is '42     '"), CharT('*'));
  check.template operator()<"answer is '{:^7d}'">(SV("answer is '  42   '"), CharT('*'));

  check.template operator()<"answer is '{:*>7d}'">(SV("answer is '*****42'"), CharT('*'));
  check.template operator()<"answer is '{:*<7d}'">(SV("answer is '42*****'"), CharT('*'));
  check.template operator()<"answer is '{:*^7d}'">(SV("answer is '**42***'"), CharT('*'));

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>07d}'">(SV("answer is '     42'"), CharT('*'));
  check.template operator()<"answer is '{:<07d}'">(SV("answer is '42     '"), CharT('*'));
  check.template operator()<"answer is '{:^07d}'">(SV("answer is '  42   '"), CharT('*'));

  // *** Sign ***
  check.template operator()<"answer is {:d}">(SV("answer is 42"), CharT('*'));
  check.template operator()<"answer is {:-d}">(SV("answer is 42"), CharT('*'));
  check.template operator()<"answer is {:+d}">(SV("answer is +42"), CharT('*'));
  check.template operator()<"answer is {: d}">(SV("answer is  42"), CharT('*'));

  // *** alternate form ***
  check.template operator()<"answer is {:+#d}">(SV("answer is +42"), CharT('*'));
  check.template operator()<"answer is {:+b}">(SV("answer is +101010"), CharT('*'));
  check.template operator()<"answer is {:+#b}">(SV("answer is +0b101010"), CharT('*'));
  check.template operator()<"answer is {:+#B}">(SV("answer is +0B101010"), CharT('*'));
  check.template operator()<"answer is {:+o}">(SV("answer is +52"), CharT('*'));
  check.template operator()<"answer is {:+#o}">(SV("answer is +052"), CharT('*'));
  check.template operator()<"answer is {:+x}">(SV("answer is +2a"), CharT('*'));
  check.template operator()<"answer is {:+#x}">(SV("answer is +0x2a"), CharT('*'));
  check.template operator()<"answer is {:+X}">(SV("answer is +2A"), CharT('*'));
  check.template operator()<"answer is {:+#X}">(SV("answer is +0X2A"), CharT('*'));

  // *** zero-padding & width ***
  check.template operator()<"answer is {:+#012d}">(SV("answer is +00000000042"), CharT('*'));
  check.template operator()<"answer is {:+012b}">(SV("answer is +00000101010"), CharT('*'));
  check.template operator()<"answer is {:+#012b}">(SV("answer is +0b000101010"), CharT('*'));
  check.template operator()<"answer is {:+#012B}">(SV("answer is +0B000101010"), CharT('*'));
  check.template operator()<"answer is {:+012o}">(SV("answer is +00000000052"), CharT('*'));
  check.template operator()<"answer is {:+#012o}">(SV("answer is +00000000052"), CharT('*'));
  check.template operator()<"answer is {:+012x}">(SV("answer is +0000000002a"), CharT('*'));
  check.template operator()<"answer is {:+#012x}">(SV("answer is +0x00000002a"), CharT('*'));
  check.template operator()<"answer is {:+012X}">(SV("answer is +0000000002A"), CharT('*'));

  check.template operator()<"answer is {:+#012X}">(SV("answer is +0X00000002A"), CharT('*'));

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.d}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.0d}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.42d}"), CharT('*'));

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
    check_exception("The format-spec type has a type not supported for a char argument", fmt, '*');
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_hex_lower_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN();                          // "nan"
  auto nan_neg = std::copysign(nan_pos, static_cast<decltype(nan_pos)>(-1.0)); // "-nan"

  // Test whether the hexadecimal letters are the proper case.
  // The precision is too large for float, so two tests are used.
  check.template operator()<"answer is '{:a}'">(SV("answer is '1.abcp+0'"), F(0x1.abcp+0));
  check.template operator()<"answer is '{:a}'">(SV("answer is '1.defp+0'"), F(0x1.defp+0));

  // *** align-fill & width ***
  check.template operator()<"answer is '{:7a}'">(SV("answer is '   1p-2'"), F(0.25));
  check.template operator()<"answer is '{:>7a}'">(SV("answer is '   1p-2'"), F(0.25));
  check.template operator()<"answer is '{:<7a}'">(SV("answer is '1p-2   '"), F(0.25));
  check.template operator()<"answer is '{:^7a}'">(SV("answer is ' 1p-2  '"), F(0.25));

  check.template operator()<"answer is '{:->7a}'">(SV("answer is '---1p-3'"), F(125e-3));
  check.template operator()<"answer is '{:-<7a}'">(SV("answer is '1p-3---'"), F(125e-3));
  check.template operator()<"answer is '{:-^7a}'">(SV("answer is '-1p-3--'"), F(125e-3));

  check.template operator()<"answer is '{:*>6a}'">(SV("answer is '***inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*<6a}'">(SV("answer is 'inf***'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*^6a}'">(SV("answer is '*inf**'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#>7a}'">(SV("answer is '###-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#<7a}'">(SV("answer is '-inf###'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#^7a}'">(SV("answer is '#-inf##'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:^>6a}'">(SV("answer is '^^^nan'"), nan_pos);
  check.template operator()<"answer is '{:^<6a}'">(SV("answer is 'nan^^^'"), nan_pos);
  check.template operator()<"answer is '{:^^6a}'">(SV("answer is '^nan^^'"), nan_pos);

  check.template operator()<"answer is '{:0>7a}'">(SV("answer is '000-nan'"), nan_neg);
  check.template operator()<"answer is '{:0<7a}'">(SV("answer is '-nan000'"), nan_neg);
  check.template operator()<"answer is '{:0^7a}'">(SV("answer is '0-nan00'"), nan_neg);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>07a}'">(SV("answer is '   1p-2'"), F(0.25));
  check.template operator()<"answer is '{:<07a}'">(SV("answer is '1p-2   '"), F(0.25));
  check.template operator()<"answer is '{:^07a}'">(SV("answer is ' 1p-2  '"), F(0.25));

  // *** Sign ***
  check.template operator()<"answer is '{:a}'">(SV("answer is '0p+0'"), F(0));
  check.template operator()<"answer is '{:-a}'">(SV("answer is '0p+0'"), F(0));
  check.template operator()<"answer is '{:+a}'">(SV("answer is '+0p+0'"), F(0));
  check.template operator()<"answer is '{: a}'">(SV("answer is ' 0p+0'"), F(0));

  check.template operator()<"answer is '{:a}'">(SV("answer is '-0p+0'"), F(-0.));
  check.template operator()<"answer is '{:-a}'">(SV("answer is '-0p+0'"), F(-0.));
  check.template operator()<"answer is '{:+a}'">(SV("answer is '-0p+0'"), F(-0.));
  check.template operator()<"answer is '{: a}'">(SV("answer is '-0p+0'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check.template operator()<"answer is '{:a}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-a}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+a}'">(SV("answer is '+inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: a}'">(SV("answer is ' inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:a}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-a}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+a}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: a}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:a}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:-a}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:+a}'">(SV("answer is '+nan'"), nan_pos);
  check.template operator()<"answer is '{: a}'">(SV("answer is ' nan'"), nan_pos);

  check.template operator()<"answer is '{:a}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:-a}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:+a}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{: a}'">(SV("answer is '-nan'"), nan_neg);

  // *** alternate form ***
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check.template operator()<"answer is '{:a}'">(SV("answer is '0p+0'"), F(0));
  check.template operator()<"answer is '{:#a}'">(SV("answer is '0.p+0'"), F(0));

  check.template operator()<"answer is '{:.0a}'">(SV("answer is '1p+1'"), F(2.5));
  check.template operator()<"answer is '{:#.0a}'">(SV("answer is '1.p+1'"), F(2.5));
  check.template operator()<"answer is '{:#a}'">(SV("answer is '1.4p+1'"), F(2.5));

  check.template operator()<"answer is '{:#a}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#a}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#a}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:#a}'">(SV("answer is '-nan'"), nan_neg);

  // *** zero-padding & width ***
  check.template operator()<"answer is '{:04a}'">(SV("answer is '1p-5'"), 0.03125);
  check.template operator()<"answer is '{:+05a}'">(SV("answer is '+1p-5'"), 0.03125);
  check.template operator()<"answer is '{:+06a}'">(SV("answer is '+01p-5'"), 0.03125);

  check.template operator()<"answer is '{:07a}'">(SV("answer is '0001p-5'"), 0.03125);
  check.template operator()<"answer is '{:-07a}'">(SV("answer is '0001p-5'"), 0.03125);
  check.template operator()<"answer is '{:+07a}'">(SV("answer is '+001p-5'"), 0.03125);
  check.template operator()<"answer is '{: 07a}'">(SV("answer is ' 001p-5'"), 0.03125);

  check.template operator()<"answer is '{:010a}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010a}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010a}'">(SV("answer is '      +inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010a}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010a}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010a}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010a}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010a}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010a}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:-010a}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:+010a}'">(SV("answer is '      +nan'"), nan_pos);
  check.template operator()<"answer is '{: 010a}'">(SV("answer is '       nan'"), nan_pos);

  check.template operator()<"answer is '{:010a}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:-010a}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:+010a}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{: 010a}'">(SV("answer is '      -nan'"), nan_neg);

  // *** precision ***
  // See format_test_floating_point_hex_lower_case_precision

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_hex_upper_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN();                          // "nan"
  auto nan_neg = std::copysign(nan_pos, static_cast<decltype(nan_pos)>(-1.0)); // "-nan"

  // Test whether the hexadecimal letters are the proper case.
  // The precision is too large for float, so two tests are used.
  check.template operator()<"answer is '{:A}'">(SV("answer is '1.ABCP+0'"), F(0x1.abcp+0));
  check.template operator()<"answer is '{:A}'">(SV("answer is '1.DEFP+0'"), F(0x1.defp+0));

  // *** align-fill & width ***
  check.template operator()<"answer is '{:7A}'">(SV("answer is '   1P-2'"), F(0.25));
  check.template operator()<"answer is '{:>7A}'">(SV("answer is '   1P-2'"), F(0.25));
  check.template operator()<"answer is '{:<7A}'">(SV("answer is '1P-2   '"), F(0.25));
  check.template operator()<"answer is '{:^7A}'">(SV("answer is ' 1P-2  '"), F(0.25));

  check.template operator()<"answer is '{:->7A}'">(SV("answer is '---1P-3'"), F(125e-3));
  check.template operator()<"answer is '{:-<7A}'">(SV("answer is '1P-3---'"), F(125e-3));
  check.template operator()<"answer is '{:-^7A}'">(SV("answer is '-1P-3--'"), F(125e-3));

  check.template operator()<"answer is '{:*>6A}'">(SV("answer is '***INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*<6A}'">(SV("answer is 'INF***'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*^6A}'">(SV("answer is '*INF**'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#>7A}'">(SV("answer is '###-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#<7A}'">(SV("answer is '-INF###'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#^7A}'">(SV("answer is '#-INF##'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:^>6A}'">(SV("answer is '^^^NAN'"), nan_pos);
  check.template operator()<"answer is '{:^<6A}'">(SV("answer is 'NAN^^^'"), nan_pos);
  check.template operator()<"answer is '{:^^6A}'">(SV("answer is '^NAN^^'"), nan_pos);

  check.template operator()<"answer is '{:0>7A}'">(SV("answer is '000-NAN'"), nan_neg);
  check.template operator()<"answer is '{:0<7A}'">(SV("answer is '-NAN000'"), nan_neg);
  check.template operator()<"answer is '{:0^7A}'">(SV("answer is '0-NAN00'"), nan_neg);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>07A}'">(SV("answer is '   1P-2'"), F(0.25));
  check.template operator()<"answer is '{:<07A}'">(SV("answer is '1P-2   '"), F(0.25));
  check.template operator()<"answer is '{:^07A}'">(SV("answer is ' 1P-2  '"), F(0.25));

  // *** Sign ***
  check.template operator()<"answer is '{:A}'">(SV("answer is '0P+0'"), F(0));
  check.template operator()<"answer is '{:-A}'">(SV("answer is '0P+0'"), F(0));
  check.template operator()<"answer is '{:+A}'">(SV("answer is '+0P+0'"), F(0));
  check.template operator()<"answer is '{: A}'">(SV("answer is ' 0P+0'"), F(0));

  check.template operator()<"answer is '{:A}'">(SV("answer is '-0P+0'"), F(-0.));
  check.template operator()<"answer is '{:-A}'">(SV("answer is '-0P+0'"), F(-0.));
  check.template operator()<"answer is '{:+A}'">(SV("answer is '-0P+0'"), F(-0.));
  check.template operator()<"answer is '{: A}'">(SV("answer is '-0P+0'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check.template operator()<"answer is '{:A}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-A}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+A}'">(SV("answer is '+INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: A}'">(SV("answer is ' INF'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:A}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-A}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+A}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: A}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:A}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:-A}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:+A}'">(SV("answer is '+NAN'"), nan_pos);
  check.template operator()<"answer is '{: A}'">(SV("answer is ' NAN'"), nan_pos);

  check.template operator()<"answer is '{:A}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{:-A}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{:+A}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{: A}'">(SV("answer is '-NAN'"), nan_neg);

  // *** alternate form ***
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check.template operator()<"answer is '{:A}'">(SV("answer is '0P+0'"), F(0));
  check.template operator()<"answer is '{:#A}'">(SV("answer is '0.P+0'"), F(0));

  check.template operator()<"answer is '{:.0A}'">(SV("answer is '1P+1'"), F(2.5));
  check.template operator()<"answer is '{:#.0A}'">(SV("answer is '1.P+1'"), F(2.5));
  check.template operator()<"answer is '{:#A}'">(SV("answer is '1.4P+1'"), F(2.5));

  check.template operator()<"answer is '{:#A}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#A}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#A}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:#A}'">(SV("answer is '-NAN'"), nan_neg);

  // *** zero-padding & width ***
  check.template operator()<"answer is '{:04A}'">(SV("answer is '1P-5'"), 0.03125);
  check.template operator()<"answer is '{:+05A}'">(SV("answer is '+1P-5'"), 0.03125);
  check.template operator()<"answer is '{:+06A}'">(SV("answer is '+01P-5'"), 0.03125);

  check.template operator()<"answer is '{:07A}'">(SV("answer is '0001P-5'"), 0.03125);
  check.template operator()<"answer is '{:-07A}'">(SV("answer is '0001P-5'"), 0.03125);
  check.template operator()<"answer is '{:+07A}'">(SV("answer is '+001P-5'"), 0.03125);
  check.template operator()<"answer is '{: 07A}'">(SV("answer is ' 001P-5'"), 0.03125);

  check.template operator()<"answer is '{:010A}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010A}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010A}'">(SV("answer is '      +INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010A}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010A}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010A}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010A}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010A}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010A}'">(SV("answer is '       NAN'"), nan_pos);
  check.template operator()<"answer is '{:-010A}'">(SV("answer is '       NAN'"), nan_pos);
  check.template operator()<"answer is '{:+010A}'">(SV("answer is '      +NAN'"), nan_pos);
  check.template operator()<"answer is '{: 010A}'">(SV("answer is '       NAN'"), nan_pos);

  check.template operator()<"answer is '{:010A}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{:-010A}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{:+010A}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{: 010A}'">(SV("answer is '      -NAN'"), nan_neg);

  // *** precision ***
  // See format_test_floating_point_hex_upper_case_precision

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_hex_lower_case_precision(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN();                          // "nan"
  auto nan_neg = std::copysign(nan_pos, static_cast<decltype(nan_pos)>(-1.0)); // "-nan"

  // *** align-fill & width ***
  check.template operator()<"answer is '{:14.6a}'">(SV("answer is '   1.000000p-2'"), F(0.25));
  check.template operator()<"answer is '{:>14.6a}'">(SV("answer is '   1.000000p-2'"), F(0.25));
  check.template operator()<"answer is '{:<14.6a}'">(SV("answer is '1.000000p-2   '"), F(0.25));
  check.template operator()<"answer is '{:^14.6a}'">(SV("answer is ' 1.000000p-2  '"), F(0.25));

  check.template operator()<"answer is '{:->14.6a}'">(SV("answer is '---1.000000p-3'"), F(125e-3));
  check.template operator()<"answer is '{:-<14.6a}'">(SV("answer is '1.000000p-3---'"), F(125e-3));
  check.template operator()<"answer is '{:-^14.6a}'">(SV("answer is '-1.000000p-3--'"), F(125e-3));

  check.template operator()<"answer is '{:*>6.6a}'">(SV("answer is '***inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*<6.6a}'">(SV("answer is 'inf***'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*^6.6a}'">(SV("answer is '*inf**'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#>7.6a}'">(SV("answer is '###-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#<7.6a}'">(SV("answer is '-inf###'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#^7.6a}'">(SV("answer is '#-inf##'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:^>6.6a}'">(SV("answer is '^^^nan'"), nan_pos);
  check.template operator()<"answer is '{:^<6.6a}'">(SV("answer is 'nan^^^'"), nan_pos);
  check.template operator()<"answer is '{:^^6.6a}'">(SV("answer is '^nan^^'"), nan_pos);

  check.template operator()<"answer is '{:0>7.6a}'">(SV("answer is '000-nan'"), nan_neg);
  check.template operator()<"answer is '{:0<7.6a}'">(SV("answer is '-nan000'"), nan_neg);
  check.template operator()<"answer is '{:0^7.6a}'">(SV("answer is '0-nan00'"), nan_neg);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>014.6a}'">(SV("answer is '   1.000000p-2'"), F(0.25));
  check.template operator()<"answer is '{:<014.6a}'">(SV("answer is '1.000000p-2   '"), F(0.25));
  check.template operator()<"answer is '{:^014.6a}'">(SV("answer is ' 1.000000p-2  '"), F(0.25));

  // *** Sign ***
  check.template operator()<"answer is '{:.6a}'">(SV("answer is '0.000000p+0'"), F(0));
  check.template operator()<"answer is '{:-.6a}'">(SV("answer is '0.000000p+0'"), F(0));
  check.template operator()<"answer is '{:+.6a}'">(SV("answer is '+0.000000p+0'"), F(0));
  check.template operator()<"answer is '{: .6a}'">(SV("answer is ' 0.000000p+0'"), F(0));

  check.template operator()<"answer is '{:.6a}'">(SV("answer is '-0.000000p+0'"), F(-0.));
  check.template operator()<"answer is '{:-.6a}'">(SV("answer is '-0.000000p+0'"), F(-0.));
  check.template operator()<"answer is '{:+.6a}'">(SV("answer is '-0.000000p+0'"), F(-0.));
  check.template operator()<"answer is '{: .6a}'">(SV("answer is '-0.000000p+0'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check.template operator()<"answer is '{:.6a}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-.6a}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+.6a}'">(SV("answer is '+inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: .6a}'">(SV("answer is ' inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:.6a}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-.6a}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+.6a}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: .6a}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:.6a}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:-.6a}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:+.6a}'">(SV("answer is '+nan'"), nan_pos);
  check.template operator()<"answer is '{: .6a}'">(SV("answer is ' nan'"), nan_pos);

  check.template operator()<"answer is '{:.6a}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:-.6a}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:+.6a}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{: .6a}'">(SV("answer is '-nan'"), nan_neg);

  // *** alternate form ***
  check.template operator()<"answer is '{:#.6a}'">(SV("answer is '1.400000p+1'"), F(2.5));

  check.template operator()<"answer is '{:#.6a}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#.6a}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#.6a}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:#.6a}'">(SV("answer is '-nan'"), nan_neg);

  // *** zero-padding & width ***
  check.template operator()<"answer is '{:011.6a}'">(SV("answer is '1.000000p-5'"), 0.03125);
  check.template operator()<"answer is '{:+012.6a}'">(SV("answer is '+1.000000p-5'"), 0.03125);
  check.template operator()<"answer is '{:+013.6a}'">(SV("answer is '+01.000000p-5'"), 0.03125);

  check.template operator()<"answer is '{:014.6a}'">(SV("answer is '0001.000000p-5'"), 0.03125);
  check.template operator()<"answer is '{:-014.6a}'">(SV("answer is '0001.000000p-5'"), 0.03125);
  check.template operator()<"answer is '{:+014.6a}'">(SV("answer is '+001.000000p-5'"), 0.03125);
  check.template operator()<"answer is '{: 014.6a}'">(SV("answer is ' 001.000000p-5'"), 0.03125);

  check.template operator()<"answer is '{:010.6a}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010.6a}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010.6a}'">(SV("answer is '      +inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010.6a}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010.6a}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010.6a}'">(SV("answer is '      -inf'"),
                                                      -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010.6a}'">(SV("answer is '      -inf'"),
                                                      -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010.6a}'">(SV("answer is '      -inf'"),
                                                      -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010.6a}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:-010.6a}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:+010.6a}'">(SV("answer is '      +nan'"), nan_pos);
  check.template operator()<"answer is '{: 010.6a}'">(SV("answer is '       nan'"), nan_pos);

  check.template operator()<"answer is '{:010.6a}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:-010.6a}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:+010.6a}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{: 010.6a}'">(SV("answer is '      -nan'"), nan_neg);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_hex_upper_case_precision(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN();                          // "nan"
  auto nan_neg = std::copysign(nan_pos, static_cast<decltype(nan_pos)>(-1.0)); // "-nan"

  // *** align-fill & width ***
  check.template operator()<"answer is '{:14.6A}'">(SV("answer is '   1.000000P-2'"), F(0.25));
  check.template operator()<"answer is '{:>14.6A}'">(SV("answer is '   1.000000P-2'"), F(0.25));
  check.template operator()<"answer is '{:<14.6A}'">(SV("answer is '1.000000P-2   '"), F(0.25));
  check.template operator()<"answer is '{:^14.6A}'">(SV("answer is ' 1.000000P-2  '"), F(0.25));

  check.template operator()<"answer is '{:->14.6A}'">(SV("answer is '---1.000000P-3'"), F(125e-3));
  check.template operator()<"answer is '{:-<14.6A}'">(SV("answer is '1.000000P-3---'"), F(125e-3));
  check.template operator()<"answer is '{:-^14.6A}'">(SV("answer is '-1.000000P-3--'"), F(125e-3));

  check.template operator()<"answer is '{:*>6.6A}'">(SV("answer is '***INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*<6.6A}'">(SV("answer is 'INF***'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*^6.6A}'">(SV("answer is '*INF**'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#>7.6A}'">(SV("answer is '###-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#<7.6A}'">(SV("answer is '-INF###'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#^7.6A}'">(SV("answer is '#-INF##'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:^>6.6A}'">(SV("answer is '^^^NAN'"), nan_pos);
  check.template operator()<"answer is '{:^<6.6A}'">(SV("answer is 'NAN^^^'"), nan_pos);
  check.template operator()<"answer is '{:^^6.6A}'">(SV("answer is '^NAN^^'"), nan_pos);

  check.template operator()<"answer is '{:0>7.6A}'">(SV("answer is '000-NAN'"), nan_neg);
  check.template operator()<"answer is '{:0<7.6A}'">(SV("answer is '-NAN000'"), nan_neg);
  check.template operator()<"answer is '{:0^7.6A}'">(SV("answer is '0-NAN00'"), nan_neg);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>014.6A}'">(SV("answer is '   1.000000P-2'"), F(0.25));
  check.template operator()<"answer is '{:<014.6A}'">(SV("answer is '1.000000P-2   '"), F(0.25));
  check.template operator()<"answer is '{:^014.6A}'">(SV("answer is ' 1.000000P-2  '"), F(0.25));

  // *** Sign ***
  check.template operator()<"answer is '{:.6A}'">(SV("answer is '0.000000P+0'"), F(0));
  check.template operator()<"answer is '{:-.6A}'">(SV("answer is '0.000000P+0'"), F(0));
  check.template operator()<"answer is '{:+.6A}'">(SV("answer is '+0.000000P+0'"), F(0));
  check.template operator()<"answer is '{: .6A}'">(SV("answer is ' 0.000000P+0'"), F(0));

  check.template operator()<"answer is '{:.6A}'">(SV("answer is '-0.000000P+0'"), F(-0.));
  check.template operator()<"answer is '{:-.6A}'">(SV("answer is '-0.000000P+0'"), F(-0.));
  check.template operator()<"answer is '{:+.6A}'">(SV("answer is '-0.000000P+0'"), F(-0.));
  check.template operator()<"answer is '{: .6A}'">(SV("answer is '-0.000000P+0'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check.template operator()<"answer is '{:.6A}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-.6A}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+.6A}'">(SV("answer is '+INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: .6A}'">(SV("answer is ' INF'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:.6A}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-.6A}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+.6A}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: .6A}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:.6A}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:-.6A}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:+.6A}'">(SV("answer is '+NAN'"), nan_pos);
  check.template operator()<"answer is '{: .6A}'">(SV("answer is ' NAN'"), nan_pos);

  check.template operator()<"answer is '{:.6A}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{:-.6A}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{:+.6A}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{: .6A}'">(SV("answer is '-NAN'"), nan_neg);

  // *** alternate form ***
  check.template operator()<"answer is '{:#.6A}'">(SV("answer is '1.400000P+1'"), F(2.5));

  check.template operator()<"answer is '{:#.6A}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#.6A}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#.6A}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:#.6A}'">(SV("answer is '-NAN'"), nan_neg);

  // *** zero-padding & width ***
  check.template operator()<"answer is '{:011.6A}'">(SV("answer is '1.000000P-5'"), 0.03125);
  check.template operator()<"answer is '{:+012.6A}'">(SV("answer is '+1.000000P-5'"), 0.03125);
  check.template operator()<"answer is '{:+013.6A}'">(SV("answer is '+01.000000P-5'"), 0.03125);

  check.template operator()<"answer is '{:014.6A}'">(SV("answer is '0001.000000P-5'"), 0.03125);
  check.template operator()<"answer is '{:-014.6A}'">(SV("answer is '0001.000000P-5'"), 0.03125);
  check.template operator()<"answer is '{:+014.6A}'">(SV("answer is '+001.000000P-5'"), 0.03125);
  check.template operator()<"answer is '{: 014.6A}'">(SV("answer is ' 001.000000P-5'"), 0.03125);

  check.template operator()<"answer is '{:010.6A}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010.6A}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010.6A}'">(SV("answer is '      +INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010.6A}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010.6A}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010.6A}'">(SV("answer is '      -INF'"),
                                                      -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010.6A}'">(SV("answer is '      -INF'"),
                                                      -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010.6A}'">(SV("answer is '      -INF'"),
                                                      -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010.6A}'">(SV("answer is '       NAN'"), nan_pos);
  check.template operator()<"answer is '{:-010.6A}'">(SV("answer is '       NAN'"), nan_pos);
  check.template operator()<"answer is '{:+010.6A}'">(SV("answer is '      +NAN'"), nan_pos);
  check.template operator()<"answer is '{: 010.6A}'">(SV("answer is '       NAN'"), nan_pos);

  check.template operator()<"answer is '{:010.6A}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{:-010.6A}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{:+010.6A}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{: 010.6A}'">(SV("answer is '      -NAN'"), nan_neg);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_scientific_lower_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN();                          // "nan"
  auto nan_neg = std::copysign(nan_pos, static_cast<decltype(nan_pos)>(-1.0)); // "-nan"

  // *** align-fill & width ***
  check.template operator()<"answer is '{:15e}'">(SV("answer is '   2.500000e-01'"), F(0.25));
  check.template operator()<"answer is '{:>15e}'">(SV("answer is '   2.500000e-01'"), F(0.25));
  check.template operator()<"answer is '{:<15e}'">(SV("answer is '2.500000e-01   '"), F(0.25));
  check.template operator()<"answer is '{:^15e}'">(SV("answer is ' 2.500000e-01  '"), F(0.25));

  check.template operator()<"answer is '{:->15e}'">(SV("answer is '---1.250000e-01'"), F(125e-3));
  check.template operator()<"answer is '{:-<15e}'">(SV("answer is '1.250000e-01---'"), F(125e-3));
  check.template operator()<"answer is '{:-^15e}'">(SV("answer is '-1.250000e-01--'"), F(125e-3));

  check.template operator()<"answer is '{:*>6e}'">(SV("answer is '***inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*<6e}'">(SV("answer is 'inf***'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*^6e}'">(SV("answer is '*inf**'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#>7e}'">(SV("answer is '###-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#<7e}'">(SV("answer is '-inf###'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#^7e}'">(SV("answer is '#-inf##'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:^>6e}'">(SV("answer is '^^^nan'"), nan_pos);
  check.template operator()<"answer is '{:^<6e}'">(SV("answer is 'nan^^^'"), nan_pos);
  check.template operator()<"answer is '{:^^6e}'">(SV("answer is '^nan^^'"), nan_pos);

  check.template operator()<"answer is '{:0>7e}'">(SV("answer is '000-nan'"), nan_neg);
  check.template operator()<"answer is '{:0<7e}'">(SV("answer is '-nan000'"), nan_neg);
  check.template operator()<"answer is '{:0^7e}'">(SV("answer is '0-nan00'"), nan_neg);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>015e}'">(SV("answer is '   2.500000e-01'"), F(0.25));
  check.template operator()<"answer is '{:<015e}'">(SV("answer is '2.500000e-01   '"), F(0.25));
  check.template operator()<"answer is '{:^015e}'">(SV("answer is ' 2.500000e-01  '"), F(0.25));

  // *** Sign ***
  check.template operator()<"answer is '{:e}'">(SV("answer is '0.000000e+00'"), F(0));
  check.template operator()<"answer is '{:-e}'">(SV("answer is '0.000000e+00'"), F(0));
  check.template operator()<"answer is '{:+e}'">(SV("answer is '+0.000000e+00'"), F(0));
  check.template operator()<"answer is '{: e}'">(SV("answer is ' 0.000000e+00'"), F(0));

  check.template operator()<"answer is '{:e}'">(SV("answer is '-0.000000e+00'"), F(-0.));
  check.template operator()<"answer is '{:-e}'">(SV("answer is '-0.000000e+00'"), F(-0.));
  check.template operator()<"answer is '{:+e}'">(SV("answer is '-0.000000e+00'"), F(-0.));
  check.template operator()<"answer is '{: e}'">(SV("answer is '-0.000000e+00'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check.template operator()<"answer is '{:e}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-e}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+e}'">(SV("answer is '+inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: e}'">(SV("answer is ' inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:e}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-e}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+e}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: e}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:e}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:-e}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:+e}'">(SV("answer is '+nan'"), nan_pos);
  check.template operator()<"answer is '{: e}'">(SV("answer is ' nan'"), nan_pos);

  check.template operator()<"answer is '{:e}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:-e}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:+e}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{: e}'">(SV("answer is '-nan'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check.template operator()<"answer is '{:.0e}'">(SV("answer is '0e+00'"), F(0));
  check.template operator()<"answer is '{:#.0e}'">(SV("answer is '0.e+00'"), F(0));

  check.template operator()<"answer is '{:#e}'">(SV("answer is '0.000000e+00'"), F(0));
  check.template operator()<"answer is '{:#e}'">(SV("answer is '2.500000e+00'"), F(2.5));

  check.template operator()<"answer is '{:#e}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#e}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#e}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:#e}'">(SV("answer is '-nan'"), nan_neg);

  // *** zero-padding & width ***
  check.template operator()<"answer is '{:07e}'">(SV("answer is '3.125000e-02'"), 0.03125);
  check.template operator()<"answer is '{:+07e}'">(SV("answer is '+3.125000e-02'"), 0.03125);
  check.template operator()<"answer is '{:+08e}'">(SV("answer is '+3.125000e-02'"), 0.03125);
  check.template operator()<"answer is '{:+09e}'">(SV("answer is '+3.125000e-02'"), 0.03125);

  check.template operator()<"answer is '{:014e}'">(SV("answer is '003.125000e-02'"), 0.03125);
  check.template operator()<"answer is '{:-014e}'">(SV("answer is '003.125000e-02'"), 0.03125);
  check.template operator()<"answer is '{:+014e}'">(SV("answer is '+03.125000e-02'"), 0.03125);
  check.template operator()<"answer is '{: 014e}'">(SV("answer is ' 03.125000e-02'"), 0.03125);

  check.template operator()<"answer is '{:010e}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010e}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010e}'">(SV("answer is '      +inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010e}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010e}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010e}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010e}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010e}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010e}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:-010e}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:+010e}'">(SV("answer is '      +nan'"), nan_pos);
  check.template operator()<"answer is '{: 010e}'">(SV("answer is '       nan'"), nan_pos);

  check.template operator()<"answer is '{:010e}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:-010e}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:+010e}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{: 010e}'">(SV("answer is '      -nan'"), nan_neg);

  // *** precision ***
  check.template operator()<"answer is '{:.0e}'">(SV("answer is '3e-02'"), 0.03125);
  check.template operator()<"answer is '{:.1e}'">(SV("answer is '3.1e-02'"), 0.03125);
  check.template operator()<"answer is '{:.3e}'">(SV("answer is '3.125e-02'"), 0.03125);
  check.template operator()<"answer is '{:.10e}'">(SV("answer is '3.1250000000e-02'"), 0.03125);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_scientific_upper_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN();                          // "nan"
  auto nan_neg = std::copysign(nan_pos, static_cast<decltype(nan_pos)>(-1.0)); // "-nan"

  // *** align-fill & width ***
  check.template operator()<"answer is '{:15E}'">(SV("answer is '   2.500000E-01'"), F(0.25));
  check.template operator()<"answer is '{:>15E}'">(SV("answer is '   2.500000E-01'"), F(0.25));
  check.template operator()<"answer is '{:<15E}'">(SV("answer is '2.500000E-01   '"), F(0.25));
  check.template operator()<"answer is '{:^15E}'">(SV("answer is ' 2.500000E-01  '"), F(0.25));

  check.template operator()<"answer is '{:->15E}'">(SV("answer is '---1.250000E-01'"), F(125e-3));
  check.template operator()<"answer is '{:-<15E}'">(SV("answer is '1.250000E-01---'"), F(125e-3));
  check.template operator()<"answer is '{:-^15E}'">(SV("answer is '-1.250000E-01--'"), F(125e-3));

  check.template operator()<"answer is '{:*>6E}'">(SV("answer is '***INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*<6E}'">(SV("answer is 'INF***'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*^6E}'">(SV("answer is '*INF**'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#>7E}'">(SV("answer is '###-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#<7E}'">(SV("answer is '-INF###'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#^7E}'">(SV("answer is '#-INF##'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:^>6E}'">(SV("answer is '^^^NAN'"), nan_pos);
  check.template operator()<"answer is '{:^<6E}'">(SV("answer is 'NAN^^^'"), nan_pos);
  check.template operator()<"answer is '{:^^6E}'">(SV("answer is '^NAN^^'"), nan_pos);

  check.template operator()<"answer is '{:0>7E}'">(SV("answer is '000-NAN'"), nan_neg);
  check.template operator()<"answer is '{:0<7E}'">(SV("answer is '-NAN000'"), nan_neg);
  check.template operator()<"answer is '{:0^7E}'">(SV("answer is '0-NAN00'"), nan_neg);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>015E}'">(SV("answer is '   2.500000E-01'"), F(0.25));
  check.template operator()<"answer is '{:<015E}'">(SV("answer is '2.500000E-01   '"), F(0.25));
  check.template operator()<"answer is '{:^015E}'">(SV("answer is ' 2.500000E-01  '"), F(0.25));

  // *** Sign ***
  check.template operator()<"answer is '{:E}'">(SV("answer is '0.000000E+00'"), F(0));
  check.template operator()<"answer is '{:-E}'">(SV("answer is '0.000000E+00'"), F(0));
  check.template operator()<"answer is '{:+E}'">(SV("answer is '+0.000000E+00'"), F(0));
  check.template operator()<"answer is '{: E}'">(SV("answer is ' 0.000000E+00'"), F(0));

  check.template operator()<"answer is '{:E}'">(SV("answer is '-0.000000E+00'"), F(-0.));
  check.template operator()<"answer is '{:-E}'">(SV("answer is '-0.000000E+00'"), F(-0.));
  check.template operator()<"answer is '{:+E}'">(SV("answer is '-0.000000E+00'"), F(-0.));
  check.template operator()<"answer is '{: E}'">(SV("answer is '-0.000000E+00'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check.template operator()<"answer is '{:E}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-E}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+E}'">(SV("answer is '+INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: E}'">(SV("answer is ' INF'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:E}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-E}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+E}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: E}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:E}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:-E}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:+E}'">(SV("answer is '+NAN'"), nan_pos);
  check.template operator()<"answer is '{: E}'">(SV("answer is ' NAN'"), nan_pos);

  check.template operator()<"answer is '{:E}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{:-E}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{:+E}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{: E}'">(SV("answer is '-NAN'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check.template operator()<"answer is '{:.0E}'">(SV("answer is '0E+00'"), F(0));
  check.template operator()<"answer is '{:#.0E}'">(SV("answer is '0.E+00'"), F(0));

  check.template operator()<"answer is '{:#E}'">(SV("answer is '0.000000E+00'"), F(0));
  check.template operator()<"answer is '{:#E}'">(SV("answer is '2.500000E+00'"), F(2.5));

  check.template operator()<"answer is '{:#E}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#E}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#E}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:#E}'">(SV("answer is '-NAN'"), nan_neg);

  // *** zero-padding & width ***
  check.template operator()<"answer is '{:07E}'">(SV("answer is '3.125000E-02'"), 0.03125);
  check.template operator()<"answer is '{:+07E}'">(SV("answer is '+3.125000E-02'"), 0.03125);
  check.template operator()<"answer is '{:+08E}'">(SV("answer is '+3.125000E-02'"), 0.03125);
  check.template operator()<"answer is '{:+09E}'">(SV("answer is '+3.125000E-02'"), 0.03125);

  check.template operator()<"answer is '{:014E}'">(SV("answer is '003.125000E-02'"), 0.03125);
  check.template operator()<"answer is '{:-014E}'">(SV("answer is '003.125000E-02'"), 0.03125);
  check.template operator()<"answer is '{:+014E}'">(SV("answer is '+03.125000E-02'"), 0.03125);
  check.template operator()<"answer is '{: 014E}'">(SV("answer is ' 03.125000E-02'"), 0.03125);

  check.template operator()<"answer is '{:010E}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010E}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010E}'">(SV("answer is '      +INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010E}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010E}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010E}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010E}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010E}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010E}'">(SV("answer is '       NAN'"), nan_pos);
  check.template operator()<"answer is '{:-010E}'">(SV("answer is '       NAN'"), nan_pos);
  check.template operator()<"answer is '{:+010E}'">(SV("answer is '      +NAN'"), nan_pos);
  check.template operator()<"answer is '{: 010E}'">(SV("answer is '       NAN'"), nan_pos);

  check.template operator()<"answer is '{:010E}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{:-010E}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{:+010E}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{: 010E}'">(SV("answer is '      -NAN'"), nan_neg);

  // *** precision ***
  check.template operator()<"answer is '{:.0E}'">(SV("answer is '3E-02'"), 0.03125);
  check.template operator()<"answer is '{:.1E}'">(SV("answer is '3.1E-02'"), 0.03125);
  check.template operator()<"answer is '{:.3E}'">(SV("answer is '3.125E-02'"), 0.03125);
  check.template operator()<"answer is '{:.10E}'">(SV("answer is '3.1250000000E-02'"), 0.03125);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_fixed_lower_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN();                          // "nan"
  auto nan_neg = std::copysign(nan_pos, static_cast<decltype(nan_pos)>(-1.0)); // "-nan"

  // *** align-fill & width ***
  check.template operator()<"answer is '{:11f}'">(SV("answer is '   0.250000'"), F(0.25));
  check.template operator()<"answer is '{:>11f}'">(SV("answer is '   0.250000'"), F(0.25));
  check.template operator()<"answer is '{:<11f}'">(SV("answer is '0.250000   '"), F(0.25));
  check.template operator()<"answer is '{:^11f}'">(SV("answer is ' 0.250000  '"), F(0.25));

  check.template operator()<"answer is '{:->11f}'">(SV("answer is '---0.125000'"), F(125e-3));
  check.template operator()<"answer is '{:-<11f}'">(SV("answer is '0.125000---'"), F(125e-3));
  check.template operator()<"answer is '{:-^11f}'">(SV("answer is '-0.125000--'"), F(125e-3));

  check.template operator()<"answer is '{:*>6f}'">(SV("answer is '***inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*<6f}'">(SV("answer is 'inf***'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*^6f}'">(SV("answer is '*inf**'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#>7f}'">(SV("answer is '###-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#<7f}'">(SV("answer is '-inf###'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#^7f}'">(SV("answer is '#-inf##'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:^>6f}'">(SV("answer is '^^^nan'"), nan_pos);
  check.template operator()<"answer is '{:^<6f}'">(SV("answer is 'nan^^^'"), nan_pos);
  check.template operator()<"answer is '{:^^6f}'">(SV("answer is '^nan^^'"), nan_pos);

  check.template operator()<"answer is '{:0>7f}'">(SV("answer is '000-nan'"), nan_neg);
  check.template operator()<"answer is '{:0<7f}'">(SV("answer is '-nan000'"), nan_neg);
  check.template operator()<"answer is '{:0^7f}'">(SV("answer is '0-nan00'"), nan_neg);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>011f}'">(SV("answer is '   0.250000'"), F(0.25));
  check.template operator()<"answer is '{:<011f}'">(SV("answer is '0.250000   '"), F(0.25));
  check.template operator()<"answer is '{:^011f}'">(SV("answer is ' 0.250000  '"), F(0.25));

  // *** Sign ***
  check.template operator()<"answer is '{:f}'">(SV("answer is '0.000000'"), F(0));
  check.template operator()<"answer is '{:-f}'">(SV("answer is '0.000000'"), F(0));
  check.template operator()<"answer is '{:+f}'">(SV("answer is '+0.000000'"), F(0));
  check.template operator()<"answer is '{: f}'">(SV("answer is ' 0.000000'"), F(0));

  check.template operator()<"answer is '{:f}'">(SV("answer is '-0.000000'"), F(-0.));
  check.template operator()<"answer is '{:-f}'">(SV("answer is '-0.000000'"), F(-0.));
  check.template operator()<"answer is '{:+f}'">(SV("answer is '-0.000000'"), F(-0.));
  check.template operator()<"answer is '{: f}'">(SV("answer is '-0.000000'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check.template operator()<"answer is '{:f}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-f}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+f}'">(SV("answer is '+inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: f}'">(SV("answer is ' inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:f}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-f}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+f}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: f}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:f}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:-f}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:+f}'">(SV("answer is '+nan'"), nan_pos);
  check.template operator()<"answer is '{: f}'">(SV("answer is ' nan'"), nan_pos);

  check.template operator()<"answer is '{:f}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:-f}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:+f}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{: f}'">(SV("answer is '-nan'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check.template operator()<"answer is '{:.0f}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:#.0f}'">(SV("answer is '0.'"), F(0));

  check.template operator()<"answer is '{:#f}'">(SV("answer is '0.000000'"), F(0));
  check.template operator()<"answer is '{:#f}'">(SV("answer is '2.500000'"), F(2.5));

  check.template operator()<"answer is '{:#f}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#f}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#f}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:#f}'">(SV("answer is '-nan'"), nan_neg);

  // *** zero-padding & width ***
  check.template operator()<"answer is '{:07f}'">(SV("answer is '0.031250'"), 0.03125);
  check.template operator()<"answer is '{:+07f}'">(SV("answer is '+0.031250'"), 0.03125);
  check.template operator()<"answer is '{:+08f}'">(SV("answer is '+0.031250'"), 0.03125);
  check.template operator()<"answer is '{:+09f}'">(SV("answer is '+0.031250'"), 0.03125);

  check.template operator()<"answer is '{:010f}'">(SV("answer is '000.031250'"), 0.03125);
  check.template operator()<"answer is '{:-010f}'">(SV("answer is '000.031250'"), 0.03125);
  check.template operator()<"answer is '{:+010f}'">(SV("answer is '+00.031250'"), 0.03125);
  check.template operator()<"answer is '{: 010f}'">(SV("answer is ' 00.031250'"), 0.03125);

  check.template operator()<"answer is '{:010f}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010f}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010f}'">(SV("answer is '      +inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010f}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010f}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010f}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010f}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010f}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010f}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:-010f}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:+010f}'">(SV("answer is '      +nan'"), nan_pos);
  check.template operator()<"answer is '{: 010f}'">(SV("answer is '       nan'"), nan_pos);

  check.template operator()<"answer is '{:010f}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:-010f}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:+010f}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{: 010f}'">(SV("answer is '      -nan'"), nan_neg);

  // *** precision ***
  check.template operator()<"answer is '{:.0f}'">(SV("answer is '0'"), 0.03125);
  check.template operator()<"answer is '{:.1f}'">(SV("answer is '0.0'"), 0.03125);
  check.template operator()<"answer is '{:.5f}'">(SV("answer is '0.03125'"), 0.03125);
  check.template operator()<"answer is '{:.10f}'">(SV("answer is '0.0312500000'"), 0.03125);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_fixed_upper_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN();                          // "nan"
  auto nan_neg = std::copysign(nan_pos, static_cast<decltype(nan_pos)>(-1.0)); // "-nan"

  // *** align-fill & width ***
  check.template operator()<"answer is '{:11F}'">(SV("answer is '   0.250000'"), F(0.25));
  check.template operator()<"answer is '{:>11F}'">(SV("answer is '   0.250000'"), F(0.25));
  check.template operator()<"answer is '{:<11F}'">(SV("answer is '0.250000   '"), F(0.25));
  check.template operator()<"answer is '{:^11F}'">(SV("answer is ' 0.250000  '"), F(0.25));

  check.template operator()<"answer is '{:->11F}'">(SV("answer is '---0.125000'"), F(125e-3));
  check.template operator()<"answer is '{:-<11F}'">(SV("answer is '0.125000---'"), F(125e-3));
  check.template operator()<"answer is '{:-^11F}'">(SV("answer is '-0.125000--'"), F(125e-3));

  check.template operator()<"answer is '{:*>6F}'">(SV("answer is '***INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*<6F}'">(SV("answer is 'INF***'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*^6F}'">(SV("answer is '*INF**'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#>7F}'">(SV("answer is '###-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#<7F}'">(SV("answer is '-INF###'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#^7F}'">(SV("answer is '#-INF##'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:^>6F}'">(SV("answer is '^^^NAN'"), nan_pos);
  check.template operator()<"answer is '{:^<6F}'">(SV("answer is 'NAN^^^'"), nan_pos);
  check.template operator()<"answer is '{:^^6F}'">(SV("answer is '^NAN^^'"), nan_pos);

  check.template operator()<"answer is '{:0>7F}'">(SV("answer is '000-NAN'"), nan_neg);
  check.template operator()<"answer is '{:0<7F}'">(SV("answer is '-NAN000'"), nan_neg);
  check.template operator()<"answer is '{:0^7F}'">(SV("answer is '0-NAN00'"), nan_neg);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>011F}'">(SV("answer is '   0.250000'"), F(0.25));
  check.template operator()<"answer is '{:<011F}'">(SV("answer is '0.250000   '"), F(0.25));
  check.template operator()<"answer is '{:^011F}'">(SV("answer is ' 0.250000  '"), F(0.25));

  // *** Sign ***
  check.template operator()<"answer is '{:F}'">(SV("answer is '0.000000'"), F(0));
  check.template operator()<"answer is '{:-F}'">(SV("answer is '0.000000'"), F(0));
  check.template operator()<"answer is '{:+F}'">(SV("answer is '+0.000000'"), F(0));
  check.template operator()<"answer is '{: F}'">(SV("answer is ' 0.000000'"), F(0));

  check.template operator()<"answer is '{:F}'">(SV("answer is '-0.000000'"), F(-0.));
  check.template operator()<"answer is '{:-F}'">(SV("answer is '-0.000000'"), F(-0.));
  check.template operator()<"answer is '{:+F}'">(SV("answer is '-0.000000'"), F(-0.));
  check.template operator()<"answer is '{: F}'">(SV("answer is '-0.000000'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check.template operator()<"answer is '{:F}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-F}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+F}'">(SV("answer is '+INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: F}'">(SV("answer is ' INF'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:F}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-F}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+F}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: F}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:F}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:-F}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:+F}'">(SV("answer is '+NAN'"), nan_pos);
  check.template operator()<"answer is '{: F}'">(SV("answer is ' NAN'"), nan_pos);

  check.template operator()<"answer is '{:F}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{:-F}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{:+F}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{: F}'">(SV("answer is '-NAN'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check.template operator()<"answer is '{:.0F}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:#.0F}'">(SV("answer is '0.'"), F(0));

  check.template operator()<"answer is '{:#F}'">(SV("answer is '0.000000'"), F(0));
  check.template operator()<"answer is '{:#F}'">(SV("answer is '2.500000'"), F(2.5));

  check.template operator()<"answer is '{:#F}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#F}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#F}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:#F}'">(SV("answer is '-NAN'"), nan_neg);

  // *** zero-padding & width ***
  check.template operator()<"answer is '{:07F}'">(SV("answer is '0.031250'"), 0.03125);
  check.template operator()<"answer is '{:+07F}'">(SV("answer is '+0.031250'"), 0.03125);
  check.template operator()<"answer is '{:+08F}'">(SV("answer is '+0.031250'"), 0.03125);
  check.template operator()<"answer is '{:+09F}'">(SV("answer is '+0.031250'"), 0.03125);

  check.template operator()<"answer is '{:010F}'">(SV("answer is '000.031250'"), 0.03125);
  check.template operator()<"answer is '{:-010F}'">(SV("answer is '000.031250'"), 0.03125);
  check.template operator()<"answer is '{:+010F}'">(SV("answer is '+00.031250'"), 0.03125);
  check.template operator()<"answer is '{: 010F}'">(SV("answer is ' 00.031250'"), 0.03125);

  check.template operator()<"answer is '{:010F}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010F}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010F}'">(SV("answer is '      +INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010F}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010F}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010F}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010F}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010F}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010F}'">(SV("answer is '       NAN'"), nan_pos);
  check.template operator()<"answer is '{:-010F}'">(SV("answer is '       NAN'"), nan_pos);
  check.template operator()<"answer is '{:+010F}'">(SV("answer is '      +NAN'"), nan_pos);
  check.template operator()<"answer is '{: 010F}'">(SV("answer is '       NAN'"), nan_pos);

  check.template operator()<"answer is '{:010F}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{:-010F}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{:+010F}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{: 010F}'">(SV("answer is '      -NAN'"), nan_neg);

  // *** precision ***
  check.template operator()<"answer is '{:.0F}'">(SV("answer is '0'"), 0.03125);
  check.template operator()<"answer is '{:.1F}'">(SV("answer is '0.0'"), 0.03125);
  check.template operator()<"answer is '{:.5F}'">(SV("answer is '0.03125'"), 0.03125);
  check.template operator()<"answer is '{:.10F}'">(SV("answer is '0.0312500000'"), 0.03125);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_general_lower_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN();                          // "nan"
  auto nan_neg = std::copysign(nan_pos, static_cast<decltype(nan_pos)>(-1.0)); // "-nan"

  // *** align-fill & width ***
  check.template operator()<"answer is '{:7g}'">(SV("answer is '   0.25'"), F(0.25));
  check.template operator()<"answer is '{:>7g}'">(SV("answer is '   0.25'"), F(0.25));
  check.template operator()<"answer is '{:<7g}'">(SV("answer is '0.25   '"), F(0.25));
  check.template operator()<"answer is '{:^7g}'">(SV("answer is ' 0.25  '"), F(0.25));

  check.template operator()<"answer is '{:->8g}'">(SV("answer is '---0.125'"), F(125e-3));
  check.template operator()<"answer is '{:-<8g}'">(SV("answer is '0.125---'"), F(125e-3));
  check.template operator()<"answer is '{:-^8g}'">(SV("answer is '-0.125--'"), F(125e-3));

  check.template operator()<"answer is '{:*>6g}'">(SV("answer is '***inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*<6g}'">(SV("answer is 'inf***'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*^6g}'">(SV("answer is '*inf**'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#>7g}'">(SV("answer is '###-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#<7g}'">(SV("answer is '-inf###'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#^7g}'">(SV("answer is '#-inf##'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:^>6g}'">(SV("answer is '^^^nan'"), nan_pos);
  check.template operator()<"answer is '{:^<6g}'">(SV("answer is 'nan^^^'"), nan_pos);
  check.template operator()<"answer is '{:^^6g}'">(SV("answer is '^nan^^'"), nan_pos);

  check.template operator()<"answer is '{:0>7g}'">(SV("answer is '000-nan'"), nan_neg);
  check.template operator()<"answer is '{:0<7g}'">(SV("answer is '-nan000'"), nan_neg);
  check.template operator()<"answer is '{:0^7g}'">(SV("answer is '0-nan00'"), nan_neg);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>07g}'">(SV("answer is '   0.25'"), F(0.25));
  check.template operator()<"answer is '{:<07g}'">(SV("answer is '0.25   '"), F(0.25));
  check.template operator()<"answer is '{:^07g}'">(SV("answer is ' 0.25  '"), F(0.25));

  // *** Sign ***
  check.template operator()<"answer is '{:g}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:-g}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:+g}'">(SV("answer is '+0'"), F(0));
  check.template operator()<"answer is '{: g}'">(SV("answer is ' 0'"), F(0));

  check.template operator()<"answer is '{:g}'">(SV("answer is '-0'"), F(-0.));
  check.template operator()<"answer is '{:-g}'">(SV("answer is '-0'"), F(-0.));
  check.template operator()<"answer is '{:+g}'">(SV("answer is '-0'"), F(-0.));
  check.template operator()<"answer is '{: g}'">(SV("answer is '-0'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check.template operator()<"answer is '{:g}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-g}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+g}'">(SV("answer is '+inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: g}'">(SV("answer is ' inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:g}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-g}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+g}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: g}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:g}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:-g}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:+g}'">(SV("answer is '+nan'"), nan_pos);
  check.template operator()<"answer is '{: g}'">(SV("answer is ' nan'"), nan_pos);

  check.template operator()<"answer is '{:g}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:-g}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:+g}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{: g}'">(SV("answer is '-nan'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check.template operator()<"answer is '{:.0g}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:#.0g}'">(SV("answer is '0.'"), F(0));

  check.template operator()<"answer is '{:#g}'">(SV("answer is '0.00000'"), F(0));
  check.template operator()<"answer is '{:#g}'">(SV("answer is '2.50000'"), F(2.5));

  check.template operator()<"answer is '{:#g}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#g}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#g}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:#g}'">(SV("answer is '-nan'"), nan_neg);

  // *** zero-padding & width ***
  check.template operator()<"answer is '{:06g}'">(SV("answer is '0.03125'"), 0.03125);
  check.template operator()<"answer is '{:+06g}'">(SV("answer is '+0.03125'"), 0.03125);
  check.template operator()<"answer is '{:+07g}'">(SV("answer is '+0.03125'"), 0.03125);
  check.template operator()<"answer is '{:+08g}'">(SV("answer is '+0.03125'"), 0.03125);

  check.template operator()<"answer is '{:09g}'">(SV("answer is '000.03125'"), 0.03125);
  check.template operator()<"answer is '{:-09g}'">(SV("answer is '000.03125'"), 0.03125);
  check.template operator()<"answer is '{:+09g}'">(SV("answer is '+00.03125'"), 0.03125);
  check.template operator()<"answer is '{: 09g}'">(SV("answer is ' 00.03125'"), 0.03125);

  check.template operator()<"answer is '{:010g}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010g}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010g}'">(SV("answer is '      +inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010g}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010g}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010g}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010g}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010g}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010g}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:-010g}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:+010g}'">(SV("answer is '      +nan'"), nan_pos);
  check.template operator()<"answer is '{: 010g}'">(SV("answer is '       nan'"), nan_pos);

  check.template operator()<"answer is '{:010g}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:-010g}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:+010g}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{: 010g}'">(SV("answer is '      -nan'"), nan_neg);

  // *** precision ***
  check.template operator()<"answer is '{:.0g}'">(SV("answer is '0.03'"), 0.03125);
  check.template operator()<"answer is '{:.1g}'">(SV("answer is '0.03'"), 0.03125);
  check.template operator()<"answer is '{:.2g}'">(SV("answer is '0.031'"), 0.03125);
  check.template operator()<"answer is '{:.3g}'">(SV("answer is '0.0312'"), 0.03125);
  check.template operator()<"answer is '{:.4g}'">(SV("answer is '0.03125'"), 0.03125);
  check.template operator()<"answer is '{:.5g}'">(SV("answer is '0.03125'"), 0.03125);
  check.template operator()<"answer is '{:.10g}'">(SV("answer is '0.03125'"), 0.03125);

  // *** precision & alternate form ***

  // Output validated with  printf("%#xg")
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.'"), 1.2, 0);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.'"), 1.2, 1);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.2'"), 1.2, 2);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.20'"), 1.2, 3);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.200'"), 1.2, 4);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.2000'"), 1.2, 5);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.20000'"), 1.2, 6);

  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.e+03'"), 1200.0, 0);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.e+03'"), 1200.0, 1);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.2e+03'"), 1200.0, 2);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.20e+03'"), 1200.0, 3);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1200.'"), 1200.0, 4);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1200.0'"), 1200.0, 5);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1200.00'"), 1200.0, 6);

  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.e+06'"), 1200000.0, 0);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.e+06'"), 1200000.0, 1);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.2e+06'"), 1200000.0, 2);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.20e+06'"), 1200000.0, 3);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.200e+06'"), 1200000.0, 4);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.2000e+06'"), 1200000.0, 5);
  check.template operator()<"answer is '{:#.{}g}'">(SV("answer is '1.20000e+06'"), 1200000.0, 6);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_general_upper_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN();                          // "nan"
  auto nan_neg = std::copysign(nan_pos, static_cast<decltype(nan_pos)>(-1.0)); // "-nan"

  // *** align-fill & width ***
  check.template operator()<"answer is '{:7G}'">(SV("answer is '   0.25'"), F(0.25));
  check.template operator()<"answer is '{:>7G}'">(SV("answer is '   0.25'"), F(0.25));
  check.template operator()<"answer is '{:<7G}'">(SV("answer is '0.25   '"), F(0.25));
  check.template operator()<"answer is '{:^7G}'">(SV("answer is ' 0.25  '"), F(0.25));

  check.template operator()<"answer is '{:->8G}'">(SV("answer is '---0.125'"), F(125e-3));
  check.template operator()<"answer is '{:-<8G}'">(SV("answer is '0.125---'"), F(125e-3));
  check.template operator()<"answer is '{:-^8G}'">(SV("answer is '-0.125--'"), F(125e-3));

  check.template operator()<"answer is '{:*>6G}'">(SV("answer is '***INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*<6G}'">(SV("answer is 'INF***'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*^6G}'">(SV("answer is '*INF**'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#>7G}'">(SV("answer is '###-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#<7G}'">(SV("answer is '-INF###'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#^7G}'">(SV("answer is '#-INF##'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:^>6G}'">(SV("answer is '^^^NAN'"), nan_pos);
  check.template operator()<"answer is '{:^<6G}'">(SV("answer is 'NAN^^^'"), nan_pos);
  check.template operator()<"answer is '{:^^6G}'">(SV("answer is '^NAN^^'"), nan_pos);

  check.template operator()<"answer is '{:0>7G}'">(SV("answer is '000-NAN'"), nan_neg);
  check.template operator()<"answer is '{:0<7G}'">(SV("answer is '-NAN000'"), nan_neg);
  check.template operator()<"answer is '{:0^7G}'">(SV("answer is '0-NAN00'"), nan_neg);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>07G}'">(SV("answer is '   0.25'"), F(0.25));
  check.template operator()<"answer is '{:<07G}'">(SV("answer is '0.25   '"), F(0.25));
  check.template operator()<"answer is '{:^07G}'">(SV("answer is ' 0.25  '"), F(0.25));

  // *** Sign ***
  check.template operator()<"answer is '{:G}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:-G}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:+G}'">(SV("answer is '+0'"), F(0));
  check.template operator()<"answer is '{: G}'">(SV("answer is ' 0'"), F(0));

  check.template operator()<"answer is '{:G}'">(SV("answer is '-0'"), F(-0.));
  check.template operator()<"answer is '{:-G}'">(SV("answer is '-0'"), F(-0.));
  check.template operator()<"answer is '{:+G}'">(SV("answer is '-0'"), F(-0.));
  check.template operator()<"answer is '{: G}'">(SV("answer is '-0'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check.template operator()<"answer is '{:G}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-G}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+G}'">(SV("answer is '+INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: G}'">(SV("answer is ' INF'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:G}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-G}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+G}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: G}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:G}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:-G}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:+G}'">(SV("answer is '+NAN'"), nan_pos);
  check.template operator()<"answer is '{: G}'">(SV("answer is ' NAN'"), nan_pos);

  check.template operator()<"answer is '{:G}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{:-G}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{:+G}'">(SV("answer is '-NAN'"), nan_neg);
  check.template operator()<"answer is '{: G}'">(SV("answer is '-NAN'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check.template operator()<"answer is '{:.0G}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:#.0G}'">(SV("answer is '0.'"), F(0));

  check.template operator()<"answer is '{:#G}'">(SV("answer is '0.00000'"), F(0));
  check.template operator()<"answer is '{:#G}'">(SV("answer is '2.50000'"), F(2.5));

  check.template operator()<"answer is '{:#G}'">(SV("answer is 'INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#G}'">(SV("answer is '-INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#G}'">(SV("answer is 'NAN'"), nan_pos);
  check.template operator()<"answer is '{:#G}'">(SV("answer is '-NAN'"), nan_neg);

  // *** zero-padding & width ***
  check.template operator()<"answer is '{:06G}'">(SV("answer is '0.03125'"), 0.03125);
  check.template operator()<"answer is '{:+06G}'">(SV("answer is '+0.03125'"), 0.03125);
  check.template operator()<"answer is '{:+07G}'">(SV("answer is '+0.03125'"), 0.03125);
  check.template operator()<"answer is '{:+08G}'">(SV("answer is '+0.03125'"), 0.03125);

  check.template operator()<"answer is '{:09G}'">(SV("answer is '000.03125'"), 0.03125);
  check.template operator()<"answer is '{:-09G}'">(SV("answer is '000.03125'"), 0.03125);
  check.template operator()<"answer is '{:+09G}'">(SV("answer is '+00.03125'"), 0.03125);
  check.template operator()<"answer is '{: 09G}'">(SV("answer is ' 00.03125'"), 0.03125);

  check.template operator()<"answer is '{:010G}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010G}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010G}'">(SV("answer is '      +INF'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010G}'">(SV("answer is '       INF'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010G}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010G}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010G}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010G}'">(SV("answer is '      -INF'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010G}'">(SV("answer is '       NAN'"), nan_pos);
  check.template operator()<"answer is '{:-010G}'">(SV("answer is '       NAN'"), nan_pos);
  check.template operator()<"answer is '{:+010G}'">(SV("answer is '      +NAN'"), nan_pos);
  check.template operator()<"answer is '{: 010G}'">(SV("answer is '       NAN'"), nan_pos);

  check.template operator()<"answer is '{:010G}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{:-010G}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{:+010G}'">(SV("answer is '      -NAN'"), nan_neg);
  check.template operator()<"answer is '{: 010G}'">(SV("answer is '      -NAN'"), nan_neg);

  // *** precision ***
  check.template operator()<"answer is '{:.0G}'">(SV("answer is '0.03'"), 0.03125);
  check.template operator()<"answer is '{:.1G}'">(SV("answer is '0.03'"), 0.03125);
  check.template operator()<"answer is '{:.2G}'">(SV("answer is '0.031'"), 0.03125);
  check.template operator()<"answer is '{:.3G}'">(SV("answer is '0.0312'"), 0.03125);
  check.template operator()<"answer is '{:.4G}'">(SV("answer is '0.03125'"), 0.03125);
  check.template operator()<"answer is '{:.5G}'">(SV("answer is '0.03125'"), 0.03125);
  check.template operator()<"answer is '{:.10G}'">(SV("answer is '0.03125'"), 0.03125);

  // *** precision & alternate form ***

  // Output validated with  printf("%#xg")
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.'"), 1.2, 0);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.'"), 1.2, 1);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.2'"), 1.2, 2);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.20'"), 1.2, 3);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.200'"), 1.2, 4);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.2000'"), 1.2, 5);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.20000'"), 1.2, 6);

  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.E+03'"), 1200.0, 0);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.E+03'"), 1200.0, 1);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.2E+03'"), 1200.0, 2);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.20E+03'"), 1200.0, 3);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1200.'"), 1200.0, 4);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1200.0'"), 1200.0, 5);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1200.00'"), 1200.0, 6);

  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.E+06'"), 1200000.0, 0);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.E+06'"), 1200000.0, 1);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.2E+06'"), 1200000.0, 2);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.20E+06'"), 1200000.0, 3);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.200E+06'"), 1200000.0, 4);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.2000E+06'"), 1200000.0, 5);
  check.template operator()<"answer is '{:#.{}G}'">(SV("answer is '1.20000E+06'"), 1200000.0, 6);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_default(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN();                          // "nan"
  auto nan_neg = std::copysign(nan_pos, static_cast<decltype(nan_pos)>(-1.0)); // "-nan"

  // *** align-fill & width ***
  check.template operator()<"answer is '{:7}'">(SV("answer is '   0.25'"), F(0.25));
  check.template operator()<"answer is '{:>7}'">(SV("answer is '   0.25'"), F(0.25));
  check.template operator()<"answer is '{:<7}'">(SV("answer is '0.25   '"), F(0.25));
  check.template operator()<"answer is '{:^7}'">(SV("answer is ' 0.25  '"), F(0.25));

  check.template operator()<"answer is '{:->8}'">(SV("answer is '---0.125'"), F(125e-3));
  check.template operator()<"answer is '{:-<8}'">(SV("answer is '0.125---'"), F(125e-3));
  check.template operator()<"answer is '{:-^8}'">(SV("answer is '-0.125--'"), F(125e-3));

  check.template operator()<"answer is '{:*>6}'">(SV("answer is '***inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*<6}'">(SV("answer is 'inf***'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*^6}'">(SV("answer is '*inf**'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#>7}'">(SV("answer is '###-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#<7}'">(SV("answer is '-inf###'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#^7}'">(SV("answer is '#-inf##'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:^>6}'">(SV("answer is '^^^nan'"), nan_pos);
  check.template operator()<"answer is '{:^<6}'">(SV("answer is 'nan^^^'"), nan_pos);
  check.template operator()<"answer is '{:^^6}'">(SV("answer is '^nan^^'"), nan_pos);

  check.template operator()<"answer is '{:0>7}'">(SV("answer is '000-nan'"), nan_neg);
  check.template operator()<"answer is '{:0<7}'">(SV("answer is '-nan000'"), nan_neg);
  check.template operator()<"answer is '{:0^7}'">(SV("answer is '0-nan00'"), nan_neg);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>07}'">(SV("answer is '   0.25'"), F(0.25));
  check.template operator()<"answer is '{:<07}'">(SV("answer is '0.25   '"), F(0.25));
  check.template operator()<"answer is '{:^07}'">(SV("answer is ' 0.25  '"), F(0.25));

  // *** Sign ***
  check.template operator()<"answer is '{:}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:-}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:+}'">(SV("answer is '+0'"), F(0));
  check.template operator()<"answer is '{: }'">(SV("answer is ' 0'"), F(0));

  check.template operator()<"answer is '{:}'">(SV("answer is '-0'"), F(-0.));
  check.template operator()<"answer is '{:-}'">(SV("answer is '-0'"), F(-0.));
  check.template operator()<"answer is '{:+}'">(SV("answer is '-0'"), F(-0.));
  check.template operator()<"answer is '{: }'">(SV("answer is '-0'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check.template operator()<"answer is '{:}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+}'">(SV("answer is '+inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: }'">(SV("answer is ' inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: }'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:-}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:+}'">(SV("answer is '+nan'"), nan_pos);
  check.template operator()<"answer is '{: }'">(SV("answer is ' nan'"), nan_pos);

  check.template operator()<"answer is '{:}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:-}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:+}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{: }'">(SV("answer is '-nan'"), nan_neg);

  // *** alternate form ***
  check.template operator()<"answer is '{:#}'">(SV("answer is '0.'"), F(0));
  check.template operator()<"answer is '{:#}'">(SV("answer is '2.5'"), F(2.5));

  check.template operator()<"answer is '{:#}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:#}'">(SV("answer is '-nan'"), nan_neg);

  // *** zero-padding & width ***
  check.template operator()<"answer is '{:07}'">(SV("answer is '0.03125'"), 0.03125);
  check.template operator()<"answer is '{:+07}'">(SV("answer is '+0.03125'"), 0.03125);
  check.template operator()<"answer is '{:+08}'">(SV("answer is '+0.03125'"), 0.03125);
  check.template operator()<"answer is '{:+09}'">(SV("answer is '+00.03125'"), 0.03125);

  check.template operator()<"answer is '{:010}'">(SV("answer is '0000.03125'"), 0.03125);
  check.template operator()<"answer is '{:-010}'">(SV("answer is '0000.03125'"), 0.03125);
  check.template operator()<"answer is '{:+010}'">(SV("answer is '+000.03125'"), 0.03125);
  check.template operator()<"answer is '{: 010}'">(SV("answer is ' 000.03125'"), 0.03125);

  check.template operator()<"answer is '{:010}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010}'">(SV("answer is '      +inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:-010}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:+010}'">(SV("answer is '      +nan'"), nan_pos);
  check.template operator()<"answer is '{: 010}'">(SV("answer is '       nan'"), nan_pos);

  check.template operator()<"answer is '{:010}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:-010}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:+010}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{: 010}'">(SV("answer is '      -nan'"), nan_neg);

  // *** precision ***
  // See format_test_floating_point_default_precision

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_default_precision(TestFunction check) {

  auto nan_pos = std::numeric_limits<F>::quiet_NaN();                          // "nan"
  auto nan_neg = std::copysign(nan_pos, static_cast<decltype(nan_pos)>(-1.0)); // "-nan"

  // *** align-fill & width ***
  check.template operator()<"answer is '{:7.6}'">(SV("answer is '   0.25'"), F(0.25));
  check.template operator()<"answer is '{:>7.6}'">(SV("answer is '   0.25'"), F(0.25));
  check.template operator()<"answer is '{:<7.6}'">(SV("answer is '0.25   '"), F(0.25));
  check.template operator()<"answer is '{:^7.6}'">(SV("answer is ' 0.25  '"), F(0.25));

  check.template operator()<"answer is '{:->8.6}'">(SV("answer is '---0.125'"), F(125e-3));
  check.template operator()<"answer is '{:-<8.6}'">(SV("answer is '0.125---'"), F(125e-3));
  check.template operator()<"answer is '{:-^8.6}'">(SV("answer is '-0.125--'"), F(125e-3));

  check.template operator()<"answer is '{:*>6.6}'">(SV("answer is '***inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*<6.6}'">(SV("answer is 'inf***'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:*^6.6}'">(SV("answer is '*inf**'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#>7.6}'">(SV("answer is '###-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#<7.6}'">(SV("answer is '-inf###'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#^7.6}'">(SV("answer is '#-inf##'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:^>6.6}'">(SV("answer is '^^^nan'"), nan_pos);
  check.template operator()<"answer is '{:^<6.6}'">(SV("answer is 'nan^^^'"), nan_pos);
  check.template operator()<"answer is '{:^^6.6}'">(SV("answer is '^nan^^'"), nan_pos);

  check.template operator()<"answer is '{:0>7.6}'">(SV("answer is '000-nan'"), nan_neg);
  check.template operator()<"answer is '{:0<7.6}'">(SV("answer is '-nan000'"), nan_neg);
  check.template operator()<"answer is '{:0^7.6}'">(SV("answer is '0-nan00'"), nan_neg);

  // Test whether zero padding is ignored
  check.template operator()<"answer is '{:>07.6}'">(SV("answer is '   0.25'"), F(0.25));
  check.template operator()<"answer is '{:<07.6}'">(SV("answer is '0.25   '"), F(0.25));
  check.template operator()<"answer is '{:^07.6}'">(SV("answer is ' 0.25  '"), F(0.25));

  // *** Sign ***
  check.template operator()<"answer is '{:.6}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:-.6}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:+.6}'">(SV("answer is '+0'"), F(0));
  check.template operator()<"answer is '{: .6}'">(SV("answer is ' 0'"), F(0));

  check.template operator()<"answer is '{:.6}'">(SV("answer is '-0'"), F(-0.));
  check.template operator()<"answer is '{:-.6}'">(SV("answer is '-0'"), F(-0.));
  check.template operator()<"answer is '{:+.6}'">(SV("answer is '-0'"), F(-0.));
  check.template operator()<"answer is '{: .6}'">(SV("answer is '-0'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check.template operator()<"answer is '{:.6}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-.6}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+.6}'">(SV("answer is '+inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: .6}'">(SV("answer is ' inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:.6}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-.6}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+.6}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: .6}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:.6}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:-.6}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:+.6}'">(SV("answer is '+nan'"), nan_pos);
  check.template operator()<"answer is '{: .6}'">(SV("answer is ' nan'"), nan_pos);

  check.template operator()<"answer is '{:.6}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:-.6}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{:+.6}'">(SV("answer is '-nan'"), nan_neg);
  check.template operator()<"answer is '{: .6}'">(SV("answer is '-nan'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check.template operator()<"answer is '{:.0}'">(SV("answer is '0'"), F(0));
  check.template operator()<"answer is '{:#.0}'">(SV("answer is '0.'"), F(0));

  check.template operator()<"answer is '{:#.6}'">(SV("answer is '0.'"), F(0));
  check.template operator()<"answer is '{:#.6}'">(SV("answer is '2.5'"), F(2.5));

  check.template operator()<"answer is '{:#.6}'">(SV("answer is 'inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:#.6}'">(SV("answer is '-inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:#.6}'">(SV("answer is 'nan'"), nan_pos);
  check.template operator()<"answer is '{:#.6}'">(SV("answer is '-nan'"), nan_neg);

  // *** zero-padding & width ***
  check.template operator()<"answer is '{:06.6}'">(SV("answer is '0.03125'"), 0.03125);
  check.template operator()<"answer is '{:+06.6}'">(SV("answer is '+0.03125'"), 0.03125);
  check.template operator()<"answer is '{:+07.6}'">(SV("answer is '+0.03125'"), 0.03125);
  check.template operator()<"answer is '{:+08.6}'">(SV("answer is '+0.03125'"), 0.03125);

  check.template operator()<"answer is '{:09.6}'">(SV("answer is '000.03125'"), 0.03125);
  check.template operator()<"answer is '{:-09.6}'">(SV("answer is '000.03125'"), 0.03125);
  check.template operator()<"answer is '{:+09.6}'">(SV("answer is '+00.03125'"), 0.03125);
  check.template operator()<"answer is '{: 09.6}'">(SV("answer is ' 00.03125'"), 0.03125);

  check.template operator()<"answer is '{:010.6}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010.6}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010.6}'">(SV("answer is '      +inf'"), std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010.6}'">(SV("answer is '       inf'"), std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010.6}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:-010.6}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{:+010.6}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());
  check.template operator()<"answer is '{: 010.6}'">(SV("answer is '      -inf'"), -std::numeric_limits<F>::infinity());

  check.template operator()<"answer is '{:010.6}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:-010.6}'">(SV("answer is '       nan'"), nan_pos);
  check.template operator()<"answer is '{:+010.6}'">(SV("answer is '      +nan'"), nan_pos);
  check.template operator()<"answer is '{: 010.6}'">(SV("answer is '       nan'"), nan_pos);

  check.template operator()<"answer is '{:010.6}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:-010.6}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{:+010.6}'">(SV("answer is '      -nan'"), nan_neg);
  check.template operator()<"answer is '{: 010.6}'">(SV("answer is '      -nan'"), nan_neg);

  // *** precision ***
  check.template operator()<"answer is '{:.0}'">(SV("answer is '0.03'"), 0.03125);
  check.template operator()<"answer is '{:.1}'">(SV("answer is '0.03'"), 0.03125);
  check.template operator()<"answer is '{:.2}'">(SV("answer is '0.031'"), 0.03125);
  check.template operator()<"answer is '{:.3}'">(SV("answer is '0.0312'"), 0.03125);
  check.template operator()<"answer is '{:.4}'">(SV("answer is '0.03125'"), 0.03125);
  check.template operator()<"answer is '{:.5}'">(SV("answer is '0.03125'"), 0.03125);
  check.template operator()<"answer is '{:.10}'">(SV("answer is '0.03125'"), 0.03125);

  // *** precision & alternate form ***

  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.'"), 1.2, 0);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.'"), 1.2, 1);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.2'"), 1.2, 2);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.2'"), 1.2, 3);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.2'"), 1.2, 4);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.2'"), 1.2, 5);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.2'"), 1.2, 6);

  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.e+03'"), 1200.0, 0);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.e+03'"), 1200.0, 1);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.2e+03'"), 1200.0, 2);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.2e+03'"), 1200.0, 3);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1200.'"), 1200.0, 4);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1200.'"), 1200.0, 5);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1200.'"), 1200.0, 6);

  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.e+06'"), 1200000.0, 0);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.e+06'"), 1200000.0, 1);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.2e+06'"), 1200000.0, 2);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.2e+06'"), 1200000.0, 3);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.2e+06'"), 1200000.0, 4);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.2e+06'"), 1200000.0, 5);
  check.template operator()<"answer is '{:#.{}}'">(SV("answer is '1.2e+06'"), 1200000.0, 6);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction, class ExceptionTest>
void format_test_floating_point(TestFunction check, ExceptionTest check_exception) {
  format_test_floating_point_hex_lower_case<F, CharT>(check);
  format_test_floating_point_hex_upper_case<F, CharT>(check);
  format_test_floating_point_hex_lower_case_precision<F, CharT>(check);
  format_test_floating_point_hex_upper_case_precision<F, CharT>(check);

  format_test_floating_point_scientific_lower_case<F, CharT>(check);
  format_test_floating_point_scientific_upper_case<F, CharT>(check);

  format_test_floating_point_fixed_lower_case<F, CharT>(check);
  format_test_floating_point_fixed_upper_case<F, CharT>(check);

  format_test_floating_point_general_lower_case<F, CharT>(check);
  format_test_floating_point_general_upper_case<F, CharT>(check);

  format_test_floating_point_default<F, CharT>(check);
  format_test_floating_point_default_precision<F, CharT>(check);

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("aAeEfFgG"))
    check_exception("The format-spec type has a type not supported for a floating-point argument", fmt, F(1));
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_floating_point(TestFunction check, ExceptionTest check_exception) {
  format_test_floating_point<float, CharT>(check, check_exception);
  format_test_floating_point<double, CharT>(check, check_exception);
  format_test_floating_point<long double, CharT>(check, check_exception);
}

template <class P, class CharT, class TestFunction, class ExceptionTest>
void format_test_pointer(TestFunction check, ExceptionTest check_exception) {
  // *** align-fill & width ***
  check.template operator()<"answer is '{:6}'">(SV("answer is '   0x0'"), P(nullptr));
  check.template operator()<"answer is '{:>6}'">(SV("answer is '   0x0'"), P(nullptr));
  check.template operator()<"answer is '{:<6}'">(SV("answer is '0x0   '"), P(nullptr));
  check.template operator()<"answer is '{:^6}'">(SV("answer is ' 0x0  '"), P(nullptr));

  check.template operator()<"answer is '{:->6}'">(SV("answer is '---0x0'"), P(nullptr));
  check.template operator()<"answer is '{:-<6}'">(SV("answer is '0x0---'"), P(nullptr));
  check.template operator()<"answer is '{:-^6}'">(SV("answer is '-0x0--'"), P(nullptr));

  // *** Sign ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:-}"), P(nullptr));
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:+}"), P(nullptr));
  check_exception("The format-spec should consume the input or end with a '}'", SV("{: }"), P(nullptr));

  // *** alternate form ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:#}"), P(nullptr));

  // *** zero-padding ***
  check_exception("A format-spec width field shouldn't have a leading zero", SV("{:0}"), P(nullptr));

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.}"), P(nullptr));

  // *** locale-specific form ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:L}"), P(nullptr));

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("p"))
    check_exception("The format-spec type has a type not supported for a pointer argument", fmt, P(nullptr));
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_handle(TestFunction check, ExceptionTest check_exception) {
  // *** Valid permuatations ***
  check.template operator()<"answer is '{}'">(SV("answer is '0xaaaa'"), status::foo);
  check.template operator()<"answer is '{:x}'">(SV("answer is '0xaaaa'"), status::foo);
  check.template operator()<"answer is '{:X}'">(SV("answer is '0XAAAA'"), status::foo);
  check.template operator()<"answer is '{:s}'">(SV("answer is 'foo'"), status::foo);

  check.template operator()<"answer is '{}'">(SV("answer is '0x5555'"), status::bar);
  check.template operator()<"answer is '{:x}'">(SV("answer is '0x5555'"), status::bar);
  check.template operator()<"answer is '{:X}'">(SV("answer is '0X5555'"), status::bar);
  check.template operator()<"answer is '{:s}'">(SV("answer is 'bar'"), status::bar);

  check.template operator()<"answer is '{}'">(SV("answer is '0xaa55'"), status::foobar);
  check.template operator()<"answer is '{:x}'">(SV("answer is '0xaa55'"), status::foobar);
  check.template operator()<"answer is '{:X}'">(SV("answer is '0XAA55'"), status::foobar);
  check.template operator()<"answer is '{:s}'">(SV("answer is 'foobar'"), status::foobar);

  // P2418 Changed the argument from a const reference to a forwarding reference.
  // This mainly affects handle classes, however since we use an abstraction
  // layer here it's "tricky" to verify whether this test would do the "right"
  // thing. So these tests are done separately.

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("xXs"))
    check_exception("The format-spec type has a type not supported for a status argument", fmt, status::foo);
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_pointer(TestFunction check, ExceptionTest check_exception) {
  format_test_pointer<std::nullptr_t, CharT>(check, check_exception);
  format_test_pointer<void*, CharT>(check, check_exception);
  format_test_pointer<const void*, CharT>(check, check_exception);
}

/// Tests special buffer functions with a "large" input.
///
/// This is a test specific for libc++, however the code should behave the same
/// on all implementations.
/// In \c __format::__output_buffer there are some special functions to optimize
/// outputting multiple characters, \c __copy, \c __transform, \c __fill. This
/// test validates whether the functions behave properly when the output size
/// doesn't fit in its internal buffer.
template <class CharT, class TestFunction>
void format_test_buffer_optimizations(TestFunction check) {
#ifdef _LIBCPP_VERSION
  // Used to validate our test sets are the proper size.
  // To test the chunked operations it needs to be larger than the internal
  // buffer. Picked a nice looking number.
  constexpr int minimum = 3 * std::__format::__internal_storage<CharT>::__buffer_size;
#else
  constexpr int minimum = 1;
#endif

  // Copy
  std::basic_string<CharT> str = STR(
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog."
      "The quick brown fox jumps over the lazy dog.");
  assert(str.size() > minimum);
  check.template operator()<"{}">(std::basic_string_view<CharT>{str}, str);

  // Fill
  std::basic_string<CharT> fill(minimum, CharT('*'));
  check.template operator()<"{:*<{}}">(std::basic_string_view<CharT>{str + fill}, str, str.size() + minimum);
  check.template operator()<"{:*^{}}">(
      std::basic_string_view<CharT>{fill + str + fill}, str, minimum + str.size() + minimum);
  check.template operator()<"{:*>{}}">(std::basic_string_view<CharT>{fill + str}, str, minimum + str.size());
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_tests(TestFunction check, ExceptionTest check_exception) {
  // *** Test escaping  ***

  check.template operator()<"{{">(SV("{"));
  check.template operator()<"}}">(SV("}"));

  // *** Test argument ID ***
  check.template operator()<"hello {0:} {1:}">(SV("hello false true"), false, true);
  check.template operator()<"hello {1:} {0:}">(SV("hello true false"), false, true);

  // *** Test many arguments ***

  // [format.args]/1
  // An instance of basic_format_args provides access to formatting arguments.
  // Implementations should optimize the representation of basic_format_args
  // for a small number of formatting arguments.
  //
  // These's no guidances what "a small number of formatting arguments" is.
  // - fmtlib uses a 15 elements
  // - libc++ uses 12 elements
  // - MSVC STL uses a different approach regardless of the number of arguments
  // - libstdc++ has no implementation yet
  // fmtlib and libc++ use a similar approach, this approach can support 16
  // elements (based on design choices both support less elements). This test
  // makes sure "the large number of formatting arguments" code path is tested.
  check.template operator()<"{}{}{}{}{}{}{}{}{}{}\t{}{}{}{}{}{}{}{}{}{}">(SV("1234567890\t1234567890"), 1, 2, 3, 4, 5,
                                                                          6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0);

  // ** Test invalid format strings ***
  check_exception("The format string terminates at a '{'", SV("{"));
  check_exception("The replacement field misses a terminating '}'", SV("{:"), 42);

  check_exception("The format string contains an invalid escape sequence", SV("}"));
  check_exception("The format string contains an invalid escape sequence", SV("{:}-}"), 42);

  check_exception("The format string contains an invalid escape sequence", SV("} "));

  check_exception("The arg-id of the format-spec starts with an invalid character", SV("{-"), 42);
  check_exception("Argument index out of bounds", SV("hello {}"));
  check_exception("Argument index out of bounds", SV("hello {0}"));
  check_exception("Argument index out of bounds", SV("hello {1}"), 42);

  // *** Test char format argument ***
  // The `char` to `wchar_t` formatting is tested separately.
  check.template operator()<"hello {}{}{}{}{}{}{}">(
      SV("hello 09azAZ!"), CharT('0'), CharT('9'), CharT('a'), CharT('z'), CharT('A'), CharT('Z'), CharT('!'));
  format_test_char<CharT>(check, check_exception);
  format_test_char_as_integer<CharT>(check, check_exception);

  // *** Test string format argument ***
  {
    CharT buffer[] = {CharT('0'), CharT('9'), CharT('a'), CharT('z'), CharT('A'), CharT('Z'), CharT('!'), 0};
    CharT* data = buffer;
    check.template operator()<"hello {}">(SV("hello 09azAZ!"), data);
  }
  {
    CharT buffer[] = {CharT('0'), CharT('9'), CharT('a'), CharT('z'), CharT('A'), CharT('Z'), CharT('!'), 0};
    const CharT* data = buffer;
    check.template operator()<"hello {}">(SV("hello 09azAZ!"), data);
  }
  {
    std::basic_string<CharT> data = STR("world");
    check.template operator()<"hello {}">(SV("hello world"), data);
  }
  {
    std::basic_string<CharT> buffer = STR("world");
    std::basic_string_view<CharT> data = buffer;
    check.template operator()<"hello {}">(SV("hello world"), data);
  }
  format_string_tests<CharT>(check, check_exception);

  // *** Test Boolean format argument ***
  check.template operator()<"hello {} {}">(SV("hello false true"), false, true);

  format_test_bool<CharT>(check, check_exception);
  format_test_bool_as_integer<CharT>(check, check_exception);

  // *** Test signed integral format argument ***
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<signed char>(42));
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<short>(42));
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<int>(42));
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<long>(42));
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<long long>(42));
#ifndef TEST_HAS_NO_INT128
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<__int128_t>(42));
#endif
  format_test_signed_integer<CharT>(check, check_exception);

  // ** Test unsigned integral format argument ***
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<unsigned char>(42));
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<unsigned short>(42));
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<unsigned>(42));
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<unsigned long>(42));
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<unsigned long long>(42));
#ifndef TEST_HAS_NO_INT128
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<__uint128_t>(42));
#endif
  format_test_unsigned_integer<CharT>(check, check_exception);

  // *** Test floating point format argument ***
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<float>(42));
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<double>(42));
  check.template operator()<"hello {}">(SV("hello 42"), static_cast<long double>(42));
  format_test_floating_point<CharT>(check, check_exception);

  // *** Test pointer formater argument ***
  check.template operator()<"hello {}">(SV("hello 0x0"), nullptr);
  check.template operator()<"hello {}">(SV("hello 0x42"), reinterpret_cast<void*>(0x42));
  check.template operator()<"hello {}">(SV("hello 0x42"), reinterpret_cast<const void*>(0x42));
  format_test_pointer<CharT>(check, check_exception);

  // *** Test handle formatter argument ***
  format_test_handle<CharT>(check, check_exception);

  // *** Test the interal buffer optimizations ***
  format_test_buffer_optimizations<CharT>(check);
}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
template <class TestFunction>
void format_tests_char_to_wchar_t(TestFunction check) {
  using CharT = wchar_t;
  check.template operator()<"hello {}{}{}{}{}">(SV("hello 09azA"), '0', '9', 'a', 'z', 'A');
}
#endif

#endif
