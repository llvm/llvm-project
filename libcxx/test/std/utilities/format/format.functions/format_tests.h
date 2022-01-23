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
#include <charconv>
#include <cmath>
#include <cstdint>

#include "make_string.h"
#include "test_macros.h"

// In this file the following template types are used:
// TestFunction must be callable as check(expected-result, string-to-format, args-to-format...)
// ExceptionTest must be callable as check_exception(expected-exception, string-to-format, args-to-format...)

#define STR(S) MAKE_STRING(CharT, S)
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

  constexpr auto parse(auto& parse_ctx) -> decltype(parse_ctx.begin()) {
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

  auto format(status s, auto& ctx) -> decltype(ctx.out()) {
    const char* names[] = {"foo", "bar", "foobar"};
    char buffer[6];
    const char* begin;
    const char* end;
    switch (type) {
    case 0:
      begin = buffer;
      buffer[0] = '0';
      buffer[1] = 'x';
      end = std::to_chars(&buffer[2], std::end(buffer), static_cast<uint16_t>(s), 16).ptr;
      break;

    case 1:
      begin = buffer;
      buffer[0] = '0';
      buffer[1] = 'X';
      end = std::to_chars(&buffer[2], std::end(buffer), static_cast<uint16_t>(s), 16).ptr;
      std::transform(static_cast<const char*>(&buffer[2]), end, &buffer[2], [](char c) { return std::toupper(c); });
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
#ifndef _LIBCPP_NO_EXCEPTIONS
    throw std::format_error(s);
#else
    (void)s;
    std::abort();
#endif
  }
};

template <class CharT>
std::vector<std::basic_string<CharT>> invalid_types(std::string valid) {
  std::vector<std::basic_string<CharT>> result;

#define CASE(T)                                                                                                        \
case #T[0]:                                                                                                            \
  result.push_back(STR("Invalid formatter type {:" #T "}"));                                                           \
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

template <class CharT, class T, class TestFunction, class ExceptionTest>
void format_test_string(T world, T universe, TestFunction check, ExceptionTest check_exception) {

  // *** Valid input tests ***
  // Unsed argument is ignored. TODO FMT what does the Standard mandate?
  check(STR("hello world"), STR("hello {}"), world, universe);
  check(STR("hello world and universe"), STR("hello {} and {}"), world, universe);
  check(STR("hello world"), STR("hello {0}"), world, universe);
  check(STR("hello universe"), STR("hello {1}"), world, universe);
  check(STR("hello universe and world"), STR("hello {1} and {0}"), world, universe);

  check(STR("hello world"), STR("hello {:_>}"), world);
  check(STR("hello    world"), STR("hello {:>8}"), world);
  check(STR("hello ___world"), STR("hello {:_>8}"), world);
  check(STR("hello _world__"), STR("hello {:_^8}"), world);
  check(STR("hello world___"), STR("hello {:_<8}"), world);

  check(STR("hello >>>world"), STR("hello {:>>8}"), world);
  check(STR("hello <<<world"), STR("hello {:<>8}"), world);
  check(STR("hello ^^^world"), STR("hello {:^>8}"), world);

  check(STR("hello $world"), STR("hello {:$>{}}"), world, 6);
  check(STR("hello $world"), STR("hello {0:$>{1}}"), world, 6);
  check(STR("hello $world"), STR("hello {1:$>{0}}"), 6, world);

  check(STR("hello world"), STR("hello {:.5}"), world);
  check(STR("hello unive"), STR("hello {:.5}"), universe);

  check(STR("hello univer"), STR("hello {:.{}}"), universe, 6);
  check(STR("hello univer"), STR("hello {0:.{1}}"), universe, 6);
  check(STR("hello univer"), STR("hello {1:.{0}}"), 6, universe);

  check(STR("hello %world%"), STR("hello {:%^7.7}"), world);
  check(STR("hello univers"), STR("hello {:%^7.7}"), universe);
  check(STR("hello %world%"), STR("hello {:%^{}.{}}"), world, 7, 7);
  check(STR("hello %world%"), STR("hello {0:%^{1}.{2}}"), world, 7, 7);
  check(STR("hello %world%"), STR("hello {0:%^{2}.{1}}"), world, 7, 7);
  check(STR("hello %world%"), STR("hello {1:%^{0}.{2}}"), 7, world, 7);

  check(STR("hello world"), STR("hello {:_>s}"), world);
  check(STR("hello $world"), STR("hello {:$>{}s}"), world, 6);
  check(STR("hello world"), STR("hello {:.5s}"), world);
  check(STR("hello univer"), STR("hello {:.{}s}"), universe, 6);
  check(STR("hello %world%"), STR("hello {:%^7.7s}"), world);

  check(STR("hello #####uni"), STR("hello {:#>8.3s}"), universe);
  check(STR("hello ##uni###"), STR("hello {:#^8.3s}"), universe);
  check(STR("hello uni#####"), STR("hello {:#<8.3s}"), universe);

  // *** sign ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("hello {:-}"), world);

  // *** alternate form ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("hello {:#}"), world);

  // *** zero-padding ***
  check_exception("A format-spec width field shouldn't have a leading zero", STR("hello {:0}"), world);

  // *** width ***
#ifdef _LIBCPP_VERSION
  // This limit isn't specified in the Standard.
  static_assert(std::__format::__number_max == 2'147'483'647, "Update the assert and the test.");
  check_exception("The numeric value of the format-spec is too large", STR("{:2147483648}"), world);
  check_exception("The numeric value of the format-spec is too large", STR("{:5000000000}"), world);
  check_exception("The numeric value of the format-spec is too large", STR("{:10000000000}"), world);
#endif

  check_exception("A format-spec width field replacement should have a positive value", STR("hello {:{}}"), world, 0);
  check_exception("A format-spec arg-id replacement shouldn't have a negative value", STR("hello {:{}}"), world, -1);
  check_exception("A format-spec arg-id replacement exceeds the maximum supported value", STR("hello {:{}}"), world,
                  unsigned(-1));
  check_exception("Argument index out of bounds", STR("hello {:{}}"), world);
  check_exception("A format-spec arg-id replacement argument isn't an integral type", STR("hello {:{}}"), world,
                  universe);
  check_exception("Using manual argument numbering in automatic argument numbering mode", STR("hello {:{0}}"), world,
                  1);
  check_exception("Using automatic argument numbering in manual argument numbering mode", STR("hello {0:{}}"), world,
                  1);
  // Arg-id may not have leading zeros.
  check_exception("Invalid arg-id", STR("hello {0:{01}}"), world, 1);

  // *** precision ***
#ifdef _LIBCPP_VERSION
  // This limit isn't specified in the Standard.
  static_assert(std::__format::__number_max == 2'147'483'647, "Update the assert and the test.");
  check_exception("The numeric value of the format-spec is too large", STR("{:.2147483648}"), world);
  check_exception("The numeric value of the format-spec is too large", STR("{:.5000000000}"), world);
  check_exception("The numeric value of the format-spec is too large", STR("{:.10000000000}"), world);
#endif

  // Precision 0 allowed, but not useful for string arguments.
  check(STR("hello "), STR("hello {:.{}}"), world, 0);
  // Precision may have leading zeros. Secondly tests the value is still base 10.
  check(STR("hello 0123456789"), STR("hello {:.000010}"), STR("0123456789abcdef"));
  check_exception("A format-spec arg-id replacement shouldn't have a negative value", STR("hello {:.{}}"), world, -1);
  check_exception("A format-spec arg-id replacement exceeds the maximum supported value", STR("hello {:.{}}"), world,
                  ~0u);
  check_exception("Argument index out of bounds", STR("hello {:.{}}"), world);
  check_exception("A format-spec arg-id replacement argument isn't an integral type", STR("hello {:.{}}"), world,
                  universe);
  check_exception("Using manual argument numbering in automatic argument numbering mode", STR("hello {:.{0}}"), world,
                  1);
  check_exception("Using automatic argument numbering in manual argument numbering mode", STR("hello {0:.{}}"), world,
                  1);
  // Arg-id may not have leading zeros.
  check_exception("Invalid arg-id", STR("hello {0:.{01}}"), world, 1);

  // *** locale-specific form ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("hello {:L}"), world);

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("s"))
    check_exception("The format-spec type has a type not supported for a string argument", fmt, world);
}

template <class CharT, class TestFunction>
void format_test_string_unicode(TestFunction check) {
  (void)check;
#ifndef TEST_HAS_NO_UNICODE
  // ß requires one column
  check(STR("aßc"), STR("{}"), STR("aßc"));

  check(STR("aßc"), STR("{:.3}"), STR("aßc"));
  check(STR("aß"), STR("{:.2}"), STR("aßc"));
  check(STR("a"), STR("{:.1}"), STR("aßc"));

  check(STR("aßc"), STR("{:3.3}"), STR("aßc"));
  check(STR("aß"), STR("{:2.2}"), STR("aßc"));
  check(STR("a"), STR("{:1.1}"), STR("aßc"));

  check(STR("aßc---"), STR("{:-<6}"), STR("aßc"));
  check(STR("-aßc--"), STR("{:-^6}"), STR("aßc"));
  check(STR("---aßc"), STR("{:->6}"), STR("aßc"));

  // \u1000 requires two columns
  check(STR("a\u1110c"), STR("{}"), STR("a\u1110c"));

  check(STR("a\u1100c"), STR("{:.4}"), STR("a\u1100c"));
  check(STR("a\u1100"), STR("{:.3}"), STR("a\u1100c"));
  check(STR("a"), STR("{:.2}"), STR("a\u1100c"));
  check(STR("a"), STR("{:.1}"), STR("a\u1100c"));

  check(STR("a\u1100c"), STR("{:-<4.4}"), STR("a\u1100c"));
  check(STR("a\u1100"), STR("{:-<3.3}"), STR("a\u1100c"));
  check(STR("a-"), STR("{:-<2.2}"), STR("a\u1100c"));
  check(STR("a"), STR("{:-<1.1}"), STR("a\u1100c"));

  check(STR("a\u1110c---"), STR("{:-<7}"), STR("a\u1110c"));
  check(STR("-a\u1110c--"), STR("{:-^7}"), STR("a\u1110c"));
  check(STR("---a\u1110c"), STR("{:->7}"), STR("a\u1110c"));
#endif // TEST_HAS_NO_UNICODE
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_string_tests(TestFunction check, ExceptionTest check_exception) {
  std::basic_string<CharT> world = STR("world");
  std::basic_string<CharT> universe = STR("universe");

  // Testing the char const[] is a bit tricky due to array to pointer decay.
  // Since there are separate tests in format.formatter.spec the array is not
  // tested here.
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
  check(STR("answer is 'true   '"), STR("answer is '{:7}'"), true);
  check(STR("answer is '   true'"), STR("answer is '{:>7}'"), true);
  check(STR("answer is 'true   '"), STR("answer is '{:<7}'"), true);
  check(STR("answer is ' true  '"), STR("answer is '{:^7}'"), true);

  check(STR("answer is 'false   '"), STR("answer is '{:8s}'"), false);
  check(STR("answer is '   false'"), STR("answer is '{:>8s}'"), false);
  check(STR("answer is 'false   '"), STR("answer is '{:<8s}'"), false);
  check(STR("answer is ' false  '"), STR("answer is '{:^8s}'"), false);

  check(STR("answer is '---true'"), STR("answer is '{:->7}'"), true);
  check(STR("answer is 'true---'"), STR("answer is '{:-<7}'"), true);
  check(STR("answer is '-true--'"), STR("answer is '{:-^7}'"), true);

  check(STR("answer is '---false'"), STR("answer is '{:->8s}'"), false);
  check(STR("answer is 'false---'"), STR("answer is '{:-<8s}'"), false);
  check(STR("answer is '-false--'"), STR("answer is '{:-^8s}'"), false);

  // *** Sign ***
  check_exception("A sign field isn't allowed in this format-spec", STR("{:-}"), true);
  check_exception("A sign field isn't allowed in this format-spec", STR("{:+}"), true);
  check_exception("A sign field isn't allowed in this format-spec", STR("{: }"), true);

  check_exception("A sign field isn't allowed in this format-spec", STR("{:-s}"), true);
  check_exception("A sign field isn't allowed in this format-spec", STR("{:+s}"), true);
  check_exception("A sign field isn't allowed in this format-spec", STR("{: s}"), true);

  // *** alternate form ***
  check_exception("An alternate form field isn't allowed in this format-spec", STR("{:#}"), true);
  check_exception("An alternate form field isn't allowed in this format-spec", STR("{:#s}"), true);

  // *** zero-padding ***
  check_exception("A zero-padding field isn't allowed in this format-spec", STR("{:0}"), true);
  check_exception("A zero-padding field isn't allowed in this format-spec", STR("{:0s}"), true);

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.0}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.42}"), true);

  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.s}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.0s}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.42s}"), true);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdosxX"))
    check_exception("The format-spec type has a type not supported for a bool argument", fmt, true);
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_bool_as_char(TestFunction check, ExceptionTest check_exception) {
  // *** align-fill & width ***
  check(STR("answer is '\1     '"), STR("answer is '{:6c}'"), true);
  check(STR("answer is '     \1'"), STR("answer is '{:>6c}'"), true);
  check(STR("answer is '\1     '"), STR("answer is '{:<6c}'"), true);
  check(STR("answer is '  \1   '"), STR("answer is '{:^6c}'"), true);

  check(STR("answer is '-----\1'"), STR("answer is '{:->6c}'"), true);
  check(STR("answer is '\1-----'"), STR("answer is '{:-<6c}'"), true);
  check(STR("answer is '--\1---'"), STR("answer is '{:-^6c}'"), true);

  check(std::basic_string<CharT>(CSTR("answer is '\0     '"), 18), STR("answer is '{:6c}'"), false);
  check(std::basic_string<CharT>(CSTR("answer is '\0     '"), 18), STR("answer is '{:6c}'"), false);
  check(std::basic_string<CharT>(CSTR("answer is '     \0'"), 18), STR("answer is '{:>6c}'"), false);
  check(std::basic_string<CharT>(CSTR("answer is '\0     '"), 18), STR("answer is '{:<6c}'"), false);
  check(std::basic_string<CharT>(CSTR("answer is '  \0   '"), 18), STR("answer is '{:^6c}'"), false);

  check(std::basic_string<CharT>(CSTR("answer is '-----\0'"), 18), STR("answer is '{:->6c}'"), false);
  check(std::basic_string<CharT>(CSTR("answer is '\0-----'"), 18), STR("answer is '{:-<6c}'"), false);
  check(std::basic_string<CharT>(CSTR("answer is '--\0---'"), 18), STR("answer is '{:-^6c}'"), false);

  // *** Sign ***
  check_exception("A sign field isn't allowed in this format-spec", STR("{:-c}"), true);
  check_exception("A sign field isn't allowed in this format-spec", STR("{:+c}"), true);
  check_exception("A sign field isn't allowed in this format-spec", STR("{: c}"), true);

  // *** alternate form ***
  check_exception("An alternate form field isn't allowed in this format-spec", STR("{:#c}"), true);

  // *** zero-padding ***
  check_exception("A zero-padding field isn't allowed in this format-spec", STR("{:0c}"), true);

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.c}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.0c}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.42c}"), true);

  // *** locale-specific form ***
  // Note it has no effect but it's allowed.
  check(STR("answer is '*'"), STR("answer is '{:Lc}'"), '*');

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdosxX"))
    check_exception("The format-spec type has a type not supported for a bool argument", fmt, true);
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_bool_as_integer(TestFunction check, ExceptionTest check_exception) {
  // *** align-fill & width ***
  check(STR("answer is '1'"), STR("answer is '{:<1d}'"), true);
  check(STR("answer is '1 '"), STR("answer is '{:<2d}'"), true);
  check(STR("answer is '0 '"), STR("answer is '{:<2d}'"), false);

  check(STR("answer is '     1'"), STR("answer is '{:6d}'"), true);
  check(STR("answer is '     1'"), STR("answer is '{:>6d}'"), true);
  check(STR("answer is '1     '"), STR("answer is '{:<6d}'"), true);
  check(STR("answer is '  1   '"), STR("answer is '{:^6d}'"), true);

  check(STR("answer is '*****0'"), STR("answer is '{:*>6d}'"), false);
  check(STR("answer is '0*****'"), STR("answer is '{:*<6d}'"), false);
  check(STR("answer is '**0***'"), STR("answer is '{:*^6d}'"), false);

  // Test whether zero padding is ignored
  check(STR("answer is '     1'"), STR("answer is '{:>06d}'"), true);
  check(STR("answer is '1     '"), STR("answer is '{:<06d}'"), true);
  check(STR("answer is '  1   '"), STR("answer is '{:^06d}'"), true);

  // *** Sign ***
  check(STR("answer is 1"), STR("answer is {:d}"), true);
  check(STR("answer is 0"), STR("answer is {:-d}"), false);
  check(STR("answer is +1"), STR("answer is {:+d}"), true);
  check(STR("answer is  0"), STR("answer is {: d}"), false);

  // *** alternate form ***
  check(STR("answer is +1"), STR("answer is {:+#d}"), true);
  check(STR("answer is +1"), STR("answer is {:+b}"), true);
  check(STR("answer is +0b1"), STR("answer is {:+#b}"), true);
  check(STR("answer is +0B1"), STR("answer is {:+#B}"), true);
  check(STR("answer is +1"), STR("answer is {:+o}"), true);
  check(STR("answer is +01"), STR("answer is {:+#o}"), true);
  check(STR("answer is +1"), STR("answer is {:+x}"), true);
  check(STR("answer is +0x1"), STR("answer is {:+#x}"), true);
  check(STR("answer is +1"), STR("answer is {:+X}"), true);
  check(STR("answer is +0X1"), STR("answer is {:+#X}"), true);

  check(STR("answer is 0"), STR("answer is {:#d}"), false);
  check(STR("answer is 0"), STR("answer is {:b}"), false);
  check(STR("answer is 0b0"), STR("answer is {:#b}"), false);
  check(STR("answer is 0B0"), STR("answer is {:#B}"), false);
  check(STR("answer is 0"), STR("answer is {:o}"), false);
  check(STR("answer is 0"), STR("answer is {:#o}"), false);
  check(STR("answer is 0"), STR("answer is {:x}"), false);
  check(STR("answer is 0x0"), STR("answer is {:#x}"), false);
  check(STR("answer is 0"), STR("answer is {:X}"), false);
  check(STR("answer is 0X0"), STR("answer is {:#X}"), false);

  // *** zero-padding & width ***
  check(STR("answer is +00000000001"), STR("answer is {:+#012d}"), true);
  check(STR("answer is +00000000001"), STR("answer is {:+012b}"), true);
  check(STR("answer is +0b000000001"), STR("answer is {:+#012b}"), true);
  check(STR("answer is +0B000000001"), STR("answer is {:+#012B}"), true);
  check(STR("answer is +00000000001"), STR("answer is {:+012o}"), true);
  check(STR("answer is +00000000001"), STR("answer is {:+#012o}"), true);
  check(STR("answer is +00000000001"), STR("answer is {:+012x}"), true);
  check(STR("answer is +0x000000001"), STR("answer is {:+#012x}"), true);
  check(STR("answer is +00000000001"), STR("answer is {:+012X}"), true);
  check(STR("answer is +0X000000001"), STR("answer is {:+#012X}"), true);

  check(STR("answer is 000000000000"), STR("answer is {:#012d}"), false);
  check(STR("answer is 000000000000"), STR("answer is {:012b}"), false);
  check(STR("answer is 0b0000000000"), STR("answer is {:#012b}"), false);
  check(STR("answer is 0B0000000000"), STR("answer is {:#012B}"), false);
  check(STR("answer is 000000000000"), STR("answer is {:012o}"), false);
  check(STR("answer is 000000000000"), STR("answer is {:#012o}"), false);
  check(STR("answer is 000000000000"), STR("answer is {:012x}"), false);
  check(STR("answer is 0x0000000000"), STR("answer is {:#012x}"), false);
  check(STR("answer is 000000000000"), STR("answer is {:012X}"), false);
  check(STR("answer is 0X0000000000"), STR("answer is {:#012X}"), false);

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.0}"), true);
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.42}"), true);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdosxX"))
    check_exception("The format-spec type has a type not supported for a bool argument", fmt, true);
}

template <class I, class CharT, class TestFunction, class ExceptionTest>
void format_test_integer_as_integer(TestFunction check, ExceptionTest check_exception) {
  // *** align-fill & width ***
  check(STR("answer is '42'"), STR("answer is '{:<1}'"), I(42));
  check(STR("answer is '42'"), STR("answer is '{:<2}'"), I(42));
  check(STR("answer is '42 '"), STR("answer is '{:<3}'"), I(42));

  check(STR("answer is '     42'"), STR("answer is '{:7}'"), I(42));
  check(STR("answer is '     42'"), STR("answer is '{:>7}'"), I(42));
  check(STR("answer is '42     '"), STR("answer is '{:<7}'"), I(42));
  check(STR("answer is '  42   '"), STR("answer is '{:^7}'"), I(42));

  check(STR("answer is '*****42'"), STR("answer is '{:*>7}'"), I(42));
  check(STR("answer is '42*****'"), STR("answer is '{:*<7}'"), I(42));
  check(STR("answer is '**42***'"), STR("answer is '{:*^7}'"), I(42));

  // Test whether zero padding is ignored
  check(STR("answer is '     42'"), STR("answer is '{:>07}'"), I(42));
  check(STR("answer is '42     '"), STR("answer is '{:<07}'"), I(42));
  check(STR("answer is '  42   '"), STR("answer is '{:^07}'"), I(42));

  // *** Sign ***
  if constexpr (std::signed_integral<I>)
    check(STR("answer is -42"), STR("answer is {}"), I(-42));
  check(STR("answer is 0"), STR("answer is {}"), I(0));
  check(STR("answer is 42"), STR("answer is {}"), I(42));

  if constexpr (std::signed_integral<I>)
    check(STR("answer is -42"), STR("answer is {:-}"), I(-42));
  check(STR("answer is 0"), STR("answer is {:-}"), I(0));
  check(STR("answer is 42"), STR("answer is {:-}"), I(42));

  if constexpr (std::signed_integral<I>)
    check(STR("answer is -42"), STR("answer is {:+}"), I(-42));
  check(STR("answer is +0"), STR("answer is {:+}"), I(0));
  check(STR("answer is +42"), STR("answer is {:+}"), I(42));

  if constexpr (std::signed_integral<I>)
    check(STR("answer is -42"), STR("answer is {: }"), I(-42));
  check(STR("answer is  0"), STR("answer is {: }"), I(0));
  check(STR("answer is  42"), STR("answer is {: }"), I(42));

  // *** alternate form ***
  if constexpr (std::signed_integral<I>) {
    check(STR("answer is -42"), STR("answer is {:#}"), I(-42));
    check(STR("answer is -42"), STR("answer is {:#d}"), I(-42));
    check(STR("answer is -101010"), STR("answer is {:b}"), I(-42));
    check(STR("answer is -0b101010"), STR("answer is {:#b}"), I(-42));
    check(STR("answer is -0B101010"), STR("answer is {:#B}"), I(-42));
    check(STR("answer is -52"), STR("answer is {:o}"), I(-42));
    check(STR("answer is -052"), STR("answer is {:#o}"), I(-42));
    check(STR("answer is -2a"), STR("answer is {:x}"), I(-42));
    check(STR("answer is -0x2a"), STR("answer is {:#x}"), I(-42));
    check(STR("answer is -2A"), STR("answer is {:X}"), I(-42));
    check(STR("answer is -0X2A"), STR("answer is {:#X}"), I(-42));
  }
  check(STR("answer is 0"), STR("answer is {:#}"), I(0));
  check(STR("answer is 0"), STR("answer is {:#d}"), I(0));
  check(STR("answer is 0"), STR("answer is {:b}"), I(0));
  check(STR("answer is 0b0"), STR("answer is {:#b}"), I(0));
  check(STR("answer is 0B0"), STR("answer is {:#B}"), I(0));
  check(STR("answer is 0"), STR("answer is {:o}"), I(0));
  check(STR("answer is 0"), STR("answer is {:#o}"), I(0));
  check(STR("answer is 0"), STR("answer is {:x}"), I(0));
  check(STR("answer is 0x0"), STR("answer is {:#x}"), I(0));
  check(STR("answer is 0"), STR("answer is {:X}"), I(0));
  check(STR("answer is 0X0"), STR("answer is {:#X}"), I(0));

  check(STR("answer is +42"), STR("answer is {:+#}"), I(42));
  check(STR("answer is +42"), STR("answer is {:+#d}"), I(42));
  check(STR("answer is +101010"), STR("answer is {:+b}"), I(42));
  check(STR("answer is +0b101010"), STR("answer is {:+#b}"), I(42));
  check(STR("answer is +0B101010"), STR("answer is {:+#B}"), I(42));
  check(STR("answer is +52"), STR("answer is {:+o}"), I(42));
  check(STR("answer is +052"), STR("answer is {:+#o}"), I(42));
  check(STR("answer is +2a"), STR("answer is {:+x}"), I(42));
  check(STR("answer is +0x2a"), STR("answer is {:+#x}"), I(42));
  check(STR("answer is +2A"), STR("answer is {:+X}"), I(42));
  check(STR("answer is +0X2A"), STR("answer is {:+#X}"), I(42));

  // *** zero-padding & width ***
  if constexpr (std::signed_integral<I>) {
    check(STR("answer is -00000000042"), STR("answer is {:#012}"), I(-42));
    check(STR("answer is -00000000042"), STR("answer is {:#012d}"), I(-42));
    check(STR("answer is -00000101010"), STR("answer is {:012b}"), I(-42));
    check(STR("answer is -0b000101010"), STR("answer is {:#012b}"), I(-42));
    check(STR("answer is -0B000101010"), STR("answer is {:#012B}"), I(-42));
    check(STR("answer is -00000000052"), STR("answer is {:012o}"), I(-42));
    check(STR("answer is -00000000052"), STR("answer is {:#012o}"), I(-42));
    check(STR("answer is -0000000002a"), STR("answer is {:012x}"), I(-42));
    check(STR("answer is -0x00000002a"), STR("answer is {:#012x}"), I(-42));
    check(STR("answer is -0000000002A"), STR("answer is {:012X}"), I(-42));
    check(STR("answer is -0X00000002A"), STR("answer is {:#012X}"), I(-42));
  }

  check(STR("answer is 000000000000"), STR("answer is {:#012}"), I(0));
  check(STR("answer is 000000000000"), STR("answer is {:#012d}"), I(0));
  check(STR("answer is 000000000000"), STR("answer is {:012b}"), I(0));
  check(STR("answer is 0b0000000000"), STR("answer is {:#012b}"), I(0));
  check(STR("answer is 0B0000000000"), STR("answer is {:#012B}"), I(0));
  check(STR("answer is 000000000000"), STR("answer is {:012o}"), I(0));
  check(STR("answer is 000000000000"), STR("answer is {:#012o}"), I(0));
  check(STR("answer is 000000000000"), STR("answer is {:012x}"), I(0));
  check(STR("answer is 0x0000000000"), STR("answer is {:#012x}"), I(0));
  check(STR("answer is 000000000000"), STR("answer is {:012X}"), I(0));
  check(STR("answer is 0X0000000000"), STR("answer is {:#012X}"), I(0));

  check(STR("answer is +00000000042"), STR("answer is {:+#012}"), I(42));
  check(STR("answer is +00000000042"), STR("answer is {:+#012d}"), I(42));
  check(STR("answer is +00000101010"), STR("answer is {:+012b}"), I(42));
  check(STR("answer is +0b000101010"), STR("answer is {:+#012b}"), I(42));
  check(STR("answer is +0B000101010"), STR("answer is {:+#012B}"), I(42));
  check(STR("answer is +00000000052"), STR("answer is {:+012o}"), I(42));
  check(STR("answer is +00000000052"), STR("answer is {:+#012o}"), I(42));
  check(STR("answer is +0000000002a"), STR("answer is {:+012x}"), I(42));
  check(STR("answer is +0x00000002a"), STR("answer is {:+#012x}"), I(42));
  check(STR("answer is +0000000002A"), STR("answer is {:+012X}"), I(42));
  check(STR("answer is +0X00000002A"), STR("answer is {:+#012X}"), I(42));

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.}"), I(0));
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.0}"), I(0));
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.42}"), I(0));

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
    check_exception("The format-spec type has a type not supported for an integer argument", fmt, 42);
}

template <class I, class CharT, class TestFunction, class ExceptionTest>
void format_test_integer_as_char(TestFunction check, ExceptionTest check_exception) {
  // *** align-fill & width ***
  check(STR("answer is '*     '"), STR("answer is '{:6c}'"), I(42));
  check(STR("answer is '     *'"), STR("answer is '{:>6c}'"), I(42));
  check(STR("answer is '*     '"), STR("answer is '{:<6c}'"), I(42));
  check(STR("answer is '  *   '"), STR("answer is '{:^6c}'"), I(42));

  check(STR("answer is '-----*'"), STR("answer is '{:->6c}'"), I(42));
  check(STR("answer is '*-----'"), STR("answer is '{:-<6c}'"), I(42));
  check(STR("answer is '--*---'"), STR("answer is '{:-^6c}'"), I(42));

  // *** Sign ***
  check(STR("answer is *"), STR("answer is {:c}"), I(42));
  check_exception("A sign field isn't allowed in this format-spec", STR("answer is {:-c}"), I(42));
  check_exception("A sign field isn't allowed in this format-spec", STR("answer is {:+c}"), I(42));
  check_exception("A sign field isn't allowed in this format-spec", STR("answer is {: c}"), I(42));

  // *** alternate form ***
  check_exception("An alternate form field isn't allowed in this format-spec", STR("answer is {:#c}"), I(42));

  // *** zero-padding & width ***
  check_exception("A zero-padding field isn't allowed in this format-spec", STR("answer is {:01c}"), I(42));

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.c}"), I(0));
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.0c}"), I(0));
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.42c}"), I(0));

  // *** locale-specific form ***
  // Note it has no effect but it's allowed.
  check(STR("answer is '*'"), STR("answer is '{:Lc}'"), I(42));

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
    check_exception("The format-spec type has a type not supported for an integer argument", fmt, I(42));

  // *** Validate range ***
  // TODO FMT Update test after adding 128-bit support.
  if constexpr (sizeof(I) <= sizeof(long long)) {
    // The code has some duplications to keep the if statement readable.
    if constexpr (std::signed_integral<CharT>) {
      if constexpr (std::signed_integral<I> && sizeof(I) > sizeof(CharT)) {
        check_exception("Integral value outside the range of the char type", STR("{:c}"),
                        std::numeric_limits<I>::min());
        check_exception("Integral value outside the range of the char type", STR("{:c}"),
                        std::numeric_limits<I>::max());
      } else if constexpr (std::unsigned_integral<I> && sizeof(I) >= sizeof(CharT)) {
        check_exception("Integral value outside the range of the char type", STR("{:c}"),
                        std::numeric_limits<I>::max());
      }
    } else if constexpr (sizeof(I) > sizeof(CharT)) {
      check_exception("Integral value outside the range of the char type", STR("{:c}"), std::numeric_limits<I>::max());
    }
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
  check(STR("-0b10000000"), STR("{:#b}"), std::numeric_limits<int8_t>::min());
  check(STR("-0200"), STR("{:#o}"), std::numeric_limits<int8_t>::min());
  check(STR("-128"), STR("{:#}"), std::numeric_limits<int8_t>::min());
  check(STR("-0x80"), STR("{:#x}"), std::numeric_limits<int8_t>::min());

  check(STR("-0b1000000000000000"), STR("{:#b}"), std::numeric_limits<int16_t>::min());
  check(STR("-0100000"), STR("{:#o}"), std::numeric_limits<int16_t>::min());
  check(STR("-32768"), STR("{:#}"), std::numeric_limits<int16_t>::min());
  check(STR("-0x8000"), STR("{:#x}"), std::numeric_limits<int16_t>::min());

  check(STR("-0b10000000000000000000000000000000"), STR("{:#b}"), std::numeric_limits<int32_t>::min());
  check(STR("-020000000000"), STR("{:#o}"), std::numeric_limits<int32_t>::min());
  check(STR("-2147483648"), STR("{:#}"), std::numeric_limits<int32_t>::min());
  check(STR("-0x80000000"), STR("{:#x}"), std::numeric_limits<int32_t>::min());

  check(STR("-0b100000000000000000000000000000000000000000000000000000000000000"
            "0"),
        STR("{:#b}"), std::numeric_limits<int64_t>::min());
  check(STR("-01000000000000000000000"), STR("{:#o}"), std::numeric_limits<int64_t>::min());
  check(STR("-9223372036854775808"), STR("{:#}"), std::numeric_limits<int64_t>::min());
  check(STR("-0x8000000000000000"), STR("{:#x}"), std::numeric_limits<int64_t>::min());

  check(STR("0b1111111"), STR("{:#b}"), std::numeric_limits<int8_t>::max());
  check(STR("0177"), STR("{:#o}"), std::numeric_limits<int8_t>::max());
  check(STR("127"), STR("{:#}"), std::numeric_limits<int8_t>::max());
  check(STR("0x7f"), STR("{:#x}"), std::numeric_limits<int8_t>::max());

  check(STR("0b111111111111111"), STR("{:#b}"), std::numeric_limits<int16_t>::max());
  check(STR("077777"), STR("{:#o}"), std::numeric_limits<int16_t>::max());
  check(STR("32767"), STR("{:#}"), std::numeric_limits<int16_t>::max());
  check(STR("0x7fff"), STR("{:#x}"), std::numeric_limits<int16_t>::max());

  check(STR("0b1111111111111111111111111111111"), STR("{:#b}"), std::numeric_limits<int32_t>::max());
  check(STR("017777777777"), STR("{:#o}"), std::numeric_limits<int32_t>::max());
  check(STR("2147483647"), STR("{:#}"), std::numeric_limits<int32_t>::max());
  check(STR("0x7fffffff"), STR("{:#x}"), std::numeric_limits<int32_t>::max());

  check(STR("0b111111111111111111111111111111111111111111111111111111111111111"), STR("{:#b}"),
        std::numeric_limits<int64_t>::max());
  check(STR("0777777777777777777777"), STR("{:#o}"), std::numeric_limits<int64_t>::max());
  check(STR("9223372036854775807"), STR("{:#}"), std::numeric_limits<int64_t>::max());
  check(STR("0x7fffffffffffffff"), STR("{:#x}"), std::numeric_limits<int64_t>::max());

  // TODO FMT Add __int128_t test after implementing full range.
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
  check(STR("0b11111111"), STR("{:#b}"), std::numeric_limits<uint8_t>::max());
  check(STR("0377"), STR("{:#o}"), std::numeric_limits<uint8_t>::max());
  check(STR("255"), STR("{:#}"), std::numeric_limits<uint8_t>::max());
  check(STR("0xff"), STR("{:#x}"), std::numeric_limits<uint8_t>::max());

  check(STR("0b1111111111111111"), STR("{:#b}"), std::numeric_limits<uint16_t>::max());
  check(STR("0177777"), STR("{:#o}"), std::numeric_limits<uint16_t>::max());
  check(STR("65535"), STR("{:#}"), std::numeric_limits<uint16_t>::max());
  check(STR("0xffff"), STR("{:#x}"), std::numeric_limits<uint16_t>::max());

  check(STR("0b11111111111111111111111111111111"), STR("{:#b}"), std::numeric_limits<uint32_t>::max());
  check(STR("037777777777"), STR("{:#o}"), std::numeric_limits<uint32_t>::max());
  check(STR("4294967295"), STR("{:#}"), std::numeric_limits<uint32_t>::max());
  check(STR("0xffffffff"), STR("{:#x}"), std::numeric_limits<uint32_t>::max());

  check(STR("0b1111111111111111111111111111111111111111111111111111111111111111"), STR("{:#b}"),
        std::numeric_limits<uint64_t>::max());
  check(STR("01777777777777777777777"), STR("{:#o}"), std::numeric_limits<uint64_t>::max());
  check(STR("18446744073709551615"), STR("{:#}"), std::numeric_limits<uint64_t>::max());
  check(STR("0xffffffffffffffff"), STR("{:#x}"), std::numeric_limits<uint64_t>::max());

  // TODO FMT Add __uint128_t test after implementing full range.
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_char(TestFunction check, ExceptionTest check_exception) {

  // ***** Char type *****
  // *** align-fill & width ***
  check(STR("answer is '*     '"), STR("answer is '{:6}'"), CharT('*'));
  check(STR("answer is '     *'"), STR("answer is '{:>6}'"), CharT('*'));
  check(STR("answer is '*     '"), STR("answer is '{:<6}'"), CharT('*'));
  check(STR("answer is '  *   '"), STR("answer is '{:^6}'"), CharT('*'));

  check(STR("answer is '*     '"), STR("answer is '{:6c}'"), CharT('*'));
  check(STR("answer is '     *'"), STR("answer is '{:>6c}'"), CharT('*'));
  check(STR("answer is '*     '"), STR("answer is '{:<6c}'"), CharT('*'));
  check(STR("answer is '  *   '"), STR("answer is '{:^6c}'"), CharT('*'));

  check(STR("answer is '-----*'"), STR("answer is '{:->6}'"), CharT('*'));
  check(STR("answer is '*-----'"), STR("answer is '{:-<6}'"), CharT('*'));
  check(STR("answer is '--*---'"), STR("answer is '{:-^6}'"), CharT('*'));

  check(STR("answer is '-----*'"), STR("answer is '{:->6c}'"), CharT('*'));
  check(STR("answer is '*-----'"), STR("answer is '{:-<6c}'"), CharT('*'));
  check(STR("answer is '--*---'"), STR("answer is '{:-^6c}'"), CharT('*'));

  // *** Sign ***
  check_exception("A sign field isn't allowed in this format-spec", STR("{:-}"), CharT('*'));
  check_exception("A sign field isn't allowed in this format-spec", STR("{:+}"), CharT('*'));
  check_exception("A sign field isn't allowed in this format-spec", STR("{: }"), CharT('*'));

  check_exception("A sign field isn't allowed in this format-spec", STR("{:-c}"), CharT('*'));
  check_exception("A sign field isn't allowed in this format-spec", STR("{:+c}"), CharT('*'));
  check_exception("A sign field isn't allowed in this format-spec", STR("{: c}"), CharT('*'));

  // *** alternate form ***
  check_exception("An alternate form field isn't allowed in this format-spec", STR("{:#}"), CharT('*'));
  check_exception("An alternate form field isn't allowed in this format-spec", STR("{:#c}"), CharT('*'));

  // *** zero-padding ***
  check_exception("A zero-padding field isn't allowed in this format-spec", STR("{:0}"), CharT('*'));
  check_exception("A zero-padding field isn't allowed in this format-spec", STR("{:0c}"), CharT('*'));

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.0}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.42}"), CharT('*'));

  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.c}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.0c}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.42c}"), CharT('*'));

  // *** locale-specific form ***
  // Note it has no effect but it's allowed.
  check(STR("answer is '*'"), STR("answer is '{:L}'"), '*');
  check(STR("answer is '*'"), STR("answer is '{:Lc}'"), '*');

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
    check_exception("The format-spec type has a type not supported for a char argument", fmt, CharT('*'));
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_char_as_integer(TestFunction check, ExceptionTest check_exception) {
  // *** align-fill & width ***
  check(STR("answer is '42'"), STR("answer is '{:<1d}'"), CharT('*'));

  check(STR("answer is '42'"), STR("answer is '{:<2d}'"), CharT('*'));
  check(STR("answer is '42 '"), STR("answer is '{:<3d}'"), CharT('*'));

  check(STR("answer is '     42'"), STR("answer is '{:7d}'"), CharT('*'));
  check(STR("answer is '     42'"), STR("answer is '{:>7d}'"), CharT('*'));
  check(STR("answer is '42     '"), STR("answer is '{:<7d}'"), CharT('*'));
  check(STR("answer is '  42   '"), STR("answer is '{:^7d}'"), CharT('*'));

  check(STR("answer is '*****42'"), STR("answer is '{:*>7d}'"), CharT('*'));
  check(STR("answer is '42*****'"), STR("answer is '{:*<7d}'"), CharT('*'));
  check(STR("answer is '**42***'"), STR("answer is '{:*^7d}'"), CharT('*'));

  // Test whether zero padding is ignored
  check(STR("answer is '     42'"), STR("answer is '{:>07d}'"), CharT('*'));
  check(STR("answer is '42     '"), STR("answer is '{:<07d}'"), CharT('*'));
  check(STR("answer is '  42   '"), STR("answer is '{:^07d}'"), CharT('*'));

  // *** Sign ***
  check(STR("answer is 42"), STR("answer is {:d}"), CharT('*'));
  check(STR("answer is 42"), STR("answer is {:-d}"), CharT('*'));
  check(STR("answer is +42"), STR("answer is {:+d}"), CharT('*'));
  check(STR("answer is  42"), STR("answer is {: d}"), CharT('*'));

  // *** alternate form ***
  check(STR("answer is +42"), STR("answer is {:+#d}"), CharT('*'));
  check(STR("answer is +101010"), STR("answer is {:+b}"), CharT('*'));
  check(STR("answer is +0b101010"), STR("answer is {:+#b}"), CharT('*'));
  check(STR("answer is +0B101010"), STR("answer is {:+#B}"), CharT('*'));
  check(STR("answer is +52"), STR("answer is {:+o}"), CharT('*'));
  check(STR("answer is +052"), STR("answer is {:+#o}"), CharT('*'));
  check(STR("answer is +2a"), STR("answer is {:+x}"), CharT('*'));
  check(STR("answer is +0x2a"), STR("answer is {:+#x}"), CharT('*'));
  check(STR("answer is +2A"), STR("answer is {:+X}"), CharT('*'));
  check(STR("answer is +0X2A"), STR("answer is {:+#X}"), CharT('*'));

  // *** zero-padding & width ***
  check(STR("answer is +00000000042"), STR("answer is {:+#012d}"), CharT('*'));
  check(STR("answer is +00000101010"), STR("answer is {:+012b}"), CharT('*'));
  check(STR("answer is +0b000101010"), STR("answer is {:+#012b}"), CharT('*'));
  check(STR("answer is +0B000101010"), STR("answer is {:+#012B}"), CharT('*'));
  check(STR("answer is +00000000052"), STR("answer is {:+012o}"), CharT('*'));
  check(STR("answer is +00000000052"), STR("answer is {:+#012o}"), CharT('*'));
  check(STR("answer is +0000000002a"), STR("answer is {:+012x}"), CharT('*'));
  check(STR("answer is +0x00000002a"), STR("answer is {:+#012x}"), CharT('*'));
  check(STR("answer is +0000000002A"), STR("answer is {:+012X}"), CharT('*'));

  check(STR("answer is +0X00000002A"), STR("answer is {:+#012X}"), CharT('*'));

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.d}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.0d}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.42d}"), CharT('*'));

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
    check_exception("The format-spec type has a type not supported for a char argument", fmt, '*');
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_hex_lower_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN(); // "nan"
  auto nan_neg = std::copysign(nan_pos, -1.0);        // "-nan"

  // Test whether the hexadecimal letters are the proper case.
  // The precision is too large for float, so two tests are used.
  check(STR("answer is '1.abcp+0'"), STR("answer is '{:a}'"), F(0x1.abcp+0));
  check(STR("answer is '1.defp+0'"), STR("answer is '{:a}'"), F(0x1.defp+0));

  // *** align-fill & width ***
  check(STR("answer is '   1p-2'"), STR("answer is '{:7a}'"), F(0.25));
  check(STR("answer is '   1p-2'"), STR("answer is '{:>7a}'"), F(0.25));
  check(STR("answer is '1p-2   '"), STR("answer is '{:<7a}'"), F(0.25));
  check(STR("answer is ' 1p-2  '"), STR("answer is '{:^7a}'"), F(0.25));

  check(STR("answer is '---1p-3'"), STR("answer is '{:->7a}'"), F(125e-3));
  check(STR("answer is '1p-3---'"), STR("answer is '{:-<7a}'"), F(125e-3));
  check(STR("answer is '-1p-3--'"), STR("answer is '{:-^7a}'"), F(125e-3));

  check(STR("answer is '***inf'"), STR("answer is '{:*>6a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf***'"), STR("answer is '{:*<6a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '*inf**'"), STR("answer is '{:*^6a}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '###-inf'"), STR("answer is '{:#>7a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf###'"), STR("answer is '{:#<7a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '#-inf##'"), STR("answer is '{:#^7a}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '^^^nan'"), STR("answer is '{:^>6a}'"), nan_pos);
  check(STR("answer is 'nan^^^'"), STR("answer is '{:^<6a}'"), nan_pos);
  check(STR("answer is '^nan^^'"), STR("answer is '{:^^6a}'"), nan_pos);

  check(STR("answer is '000-nan'"), STR("answer is '{:0>7a}'"), nan_neg);
  check(STR("answer is '-nan000'"), STR("answer is '{:0<7a}'"), nan_neg);
  check(STR("answer is '0-nan00'"), STR("answer is '{:0^7a}'"), nan_neg);

  // Test whether zero padding is ignored
  check(STR("answer is '   1p-2'"), STR("answer is '{:>07a}'"), F(0.25));
  check(STR("answer is '1p-2   '"), STR("answer is '{:<07a}'"), F(0.25));
  check(STR("answer is ' 1p-2  '"), STR("answer is '{:^07a}'"), F(0.25));

  // *** Sign ***
  check(STR("answer is '0p+0'"), STR("answer is '{:a}'"), F(0));
  check(STR("answer is '0p+0'"), STR("answer is '{:-a}'"), F(0));
  check(STR("answer is '+0p+0'"), STR("answer is '{:+a}'"), F(0));
  check(STR("answer is ' 0p+0'"), STR("answer is '{: a}'"), F(0));

  check(STR("answer is '-0p+0'"), STR("answer is '{:a}'"), F(-0.));
  check(STR("answer is '-0p+0'"), STR("answer is '{:-a}'"), F(-0.));
  check(STR("answer is '-0p+0'"), STR("answer is '{:+a}'"), F(-0.));
  check(STR("answer is '-0p+0'"), STR("answer is '{: a}'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check(STR("answer is 'inf'"), STR("answer is '{:a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf'"), STR("answer is '{:-a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '+inf'"), STR("answer is '{:+a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is ' inf'"), STR("answer is '{: a}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '-inf'"), STR("answer is '{:a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:-a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:+a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{: a}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:a}'"), nan_pos);
  check(STR("answer is 'nan'"), STR("answer is '{:-a}'"), nan_pos);
  check(STR("answer is '+nan'"), STR("answer is '{:+a}'"), nan_pos);
  check(STR("answer is ' nan'"), STR("answer is '{: a}'"), nan_pos);

  check(STR("answer is '-nan'"), STR("answer is '{:a}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:-a}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:+a}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{: a}'"), nan_neg);

  // *** alternate form ***
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check(STR("answer is '0p+0'"), STR("answer is '{:a}'"), F(0));
  check(STR("answer is '0.p+0'"), STR("answer is '{:#a}'"), F(0));

  check(STR("answer is '1p+1'"), STR("answer is '{:.0a}'"), F(2.5));
  check(STR("answer is '1.p+1'"), STR("answer is '{:#.0a}'"), F(2.5));
  check(STR("answer is '1.4p+1'"), STR("answer is '{:#a}'"), F(2.5));

  check(STR("answer is 'inf'"), STR("answer is '{:#a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:#a}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:#a}'"), nan_pos);
  check(STR("answer is '-nan'"), STR("answer is '{:#a}'"), nan_neg);

  // *** zero-padding & width ***
  check(STR("answer is '1p-5'"), STR("answer is '{:04a}'"), 0.03125);
  check(STR("answer is '+1p-5'"), STR("answer is '{:+05a}'"), 0.03125);
  check(STR("answer is '+01p-5'"), STR("answer is '{:+06a}'"), 0.03125);

  check(STR("answer is '0001p-5'"), STR("answer is '{:07a}'"), 0.03125);
  check(STR("answer is '0001p-5'"), STR("answer is '{:-07a}'"), 0.03125);
  check(STR("answer is '+001p-5'"), STR("answer is '{:+07a}'"), 0.03125);
  check(STR("answer is ' 001p-5'"), STR("answer is '{: 07a}'"), 0.03125);

  check(STR("answer is '       inf'"), STR("answer is '{:010a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{:-010a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '      +inf'"), STR("answer is '{:+010a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{: 010a}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '      -inf'"), STR("answer is '{:010a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:-010a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:+010a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{: 010a}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '       nan'"), STR("answer is '{:010a}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{:-010a}'"), nan_pos);
  check(STR("answer is '      +nan'"), STR("answer is '{:+010a}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{: 010a}'"), nan_pos);

  check(STR("answer is '      -nan'"), STR("answer is '{:010a}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:-010a}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:+010a}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{: 010a}'"), nan_neg);

  // *** precision ***
  // See format_test_floating_point_hex_lower_case_precision

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_hex_upper_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN(); // "nan"
  auto nan_neg = std::copysign(nan_pos, -1.0);        // "-nan"

  // Test whether the hexadecimal letters are the proper case.
  // The precision is too large for float, so two tests are used.
  check(STR("answer is '1.ABCP+0'"), STR("answer is '{:A}'"), F(0x1.abcp+0));
  check(STR("answer is '1.DEFP+0'"), STR("answer is '{:A}'"), F(0x1.defp+0));

  // *** align-fill & width ***
  check(STR("answer is '   1P-2'"), STR("answer is '{:7A}'"), F(0.25));
  check(STR("answer is '   1P-2'"), STR("answer is '{:>7A}'"), F(0.25));
  check(STR("answer is '1P-2   '"), STR("answer is '{:<7A}'"), F(0.25));
  check(STR("answer is ' 1P-2  '"), STR("answer is '{:^7A}'"), F(0.25));

  check(STR("answer is '---1P-3'"), STR("answer is '{:->7A}'"), F(125e-3));
  check(STR("answer is '1P-3---'"), STR("answer is '{:-<7A}'"), F(125e-3));
  check(STR("answer is '-1P-3--'"), STR("answer is '{:-^7A}'"), F(125e-3));

  check(STR("answer is '***INF'"), STR("answer is '{:*>6A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'INF***'"), STR("answer is '{:*<6A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '*INF**'"), STR("answer is '{:*^6A}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '###-INF'"), STR("answer is '{:#>7A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF###'"), STR("answer is '{:#<7A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '#-INF##'"), STR("answer is '{:#^7A}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '^^^NAN'"), STR("answer is '{:^>6A}'"), nan_pos);
  check(STR("answer is 'NAN^^^'"), STR("answer is '{:^<6A}'"), nan_pos);
  check(STR("answer is '^NAN^^'"), STR("answer is '{:^^6A}'"), nan_pos);

  check(STR("answer is '000-NAN'"), STR("answer is '{:0>7A}'"), nan_neg);
  check(STR("answer is '-NAN000'"), STR("answer is '{:0<7A}'"), nan_neg);
  check(STR("answer is '0-NAN00'"), STR("answer is '{:0^7A}'"), nan_neg);

  // Test whether zero padding is ignored
  check(STR("answer is '   1P-2'"), STR("answer is '{:>07A}'"), F(0.25));
  check(STR("answer is '1P-2   '"), STR("answer is '{:<07A}'"), F(0.25));
  check(STR("answer is ' 1P-2  '"), STR("answer is '{:^07A}'"), F(0.25));

  // *** Sign ***
  check(STR("answer is '0P+0'"), STR("answer is '{:A}'"), F(0));
  check(STR("answer is '0P+0'"), STR("answer is '{:-A}'"), F(0));
  check(STR("answer is '+0P+0'"), STR("answer is '{:+A}'"), F(0));
  check(STR("answer is ' 0P+0'"), STR("answer is '{: A}'"), F(0));

  check(STR("answer is '-0P+0'"), STR("answer is '{:A}'"), F(-0.));
  check(STR("answer is '-0P+0'"), STR("answer is '{:-A}'"), F(-0.));
  check(STR("answer is '-0P+0'"), STR("answer is '{:+A}'"), F(-0.));
  check(STR("answer is '-0P+0'"), STR("answer is '{: A}'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check(STR("answer is 'INF'"), STR("answer is '{:A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'INF'"), STR("answer is '{:-A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '+INF'"), STR("answer is '{:+A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is ' INF'"), STR("answer is '{: A}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '-INF'"), STR("answer is '{:A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:-A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:+A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{: A}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'NAN'"), STR("answer is '{:A}'"), nan_pos);
  check(STR("answer is 'NAN'"), STR("answer is '{:-A}'"), nan_pos);
  check(STR("answer is '+NAN'"), STR("answer is '{:+A}'"), nan_pos);
  check(STR("answer is ' NAN'"), STR("answer is '{: A}'"), nan_pos);

  check(STR("answer is '-NAN'"), STR("answer is '{:A}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{:-A}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{:+A}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{: A}'"), nan_neg);

  // *** alternate form ***
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check(STR("answer is '0P+0'"), STR("answer is '{:A}'"), F(0));
  check(STR("answer is '0.P+0'"), STR("answer is '{:#A}'"), F(0));

  check(STR("answer is '1P+1'"), STR("answer is '{:.0A}'"), F(2.5));
  check(STR("answer is '1.P+1'"), STR("answer is '{:#.0A}'"), F(2.5));
  check(STR("answer is '1.4P+1'"), STR("answer is '{:#A}'"), F(2.5));

  check(STR("answer is 'INF'"), STR("answer is '{:#A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:#A}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'NAN'"), STR("answer is '{:#A}'"), nan_pos);
  check(STR("answer is '-NAN'"), STR("answer is '{:#A}'"), nan_neg);

  // *** zero-padding & width ***
  check(STR("answer is '1P-5'"), STR("answer is '{:04A}'"), 0.03125);
  check(STR("answer is '+1P-5'"), STR("answer is '{:+05A}'"), 0.03125);
  check(STR("answer is '+01P-5'"), STR("answer is '{:+06A}'"), 0.03125);

  check(STR("answer is '0001P-5'"), STR("answer is '{:07A}'"), 0.03125);
  check(STR("answer is '0001P-5'"), STR("answer is '{:-07A}'"), 0.03125);
  check(STR("answer is '+001P-5'"), STR("answer is '{:+07A}'"), 0.03125);
  check(STR("answer is ' 001P-5'"), STR("answer is '{: 07A}'"), 0.03125);

  check(STR("answer is '       INF'"), STR("answer is '{:010A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       INF'"), STR("answer is '{:-010A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '      +INF'"), STR("answer is '{:+010A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       INF'"), STR("answer is '{: 010A}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '      -INF'"), STR("answer is '{:010A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{:-010A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{:+010A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{: 010A}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '       NAN'"), STR("answer is '{:010A}'"), nan_pos);
  check(STR("answer is '       NAN'"), STR("answer is '{:-010A}'"), nan_pos);
  check(STR("answer is '      +NAN'"), STR("answer is '{:+010A}'"), nan_pos);
  check(STR("answer is '       NAN'"), STR("answer is '{: 010A}'"), nan_pos);

  check(STR("answer is '      -NAN'"), STR("answer is '{:010A}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{:-010A}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{:+010A}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{: 010A}'"), nan_neg);

  // *** precision ***
  // See format_test_floating_point_hex_upper_case_precision

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_hex_lower_case_precision(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN(); // "nan"
  auto nan_neg = std::copysign(nan_pos, -1.0);        // "-nan"

  // *** align-fill & width ***
  check(STR("answer is '   1.000000p-2'"), STR("answer is '{:14.6a}'"), F(0.25));
  check(STR("answer is '   1.000000p-2'"), STR("answer is '{:>14.6a}'"), F(0.25));
  check(STR("answer is '1.000000p-2   '"), STR("answer is '{:<14.6a}'"), F(0.25));
  check(STR("answer is ' 1.000000p-2  '"), STR("answer is '{:^14.6a}'"), F(0.25));

  check(STR("answer is '---1.000000p-3'"), STR("answer is '{:->14.6a}'"), F(125e-3));
  check(STR("answer is '1.000000p-3---'"), STR("answer is '{:-<14.6a}'"), F(125e-3));
  check(STR("answer is '-1.000000p-3--'"), STR("answer is '{:-^14.6a}'"), F(125e-3));

  check(STR("answer is '***inf'"), STR("answer is '{:*>6.6a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf***'"), STR("answer is '{:*<6.6a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '*inf**'"), STR("answer is '{:*^6.6a}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '###-inf'"), STR("answer is '{:#>7.6a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf###'"), STR("answer is '{:#<7.6a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '#-inf##'"), STR("answer is '{:#^7.6a}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '^^^nan'"), STR("answer is '{:^>6.6a}'"), nan_pos);
  check(STR("answer is 'nan^^^'"), STR("answer is '{:^<6.6a}'"), nan_pos);
  check(STR("answer is '^nan^^'"), STR("answer is '{:^^6.6a}'"), nan_pos);

  check(STR("answer is '000-nan'"), STR("answer is '{:0>7.6a}'"), nan_neg);
  check(STR("answer is '-nan000'"), STR("answer is '{:0<7.6a}'"), nan_neg);
  check(STR("answer is '0-nan00'"), STR("answer is '{:0^7.6a}'"), nan_neg);

  // Test whether zero padding is ignored
  check(STR("answer is '   1.000000p-2'"), STR("answer is '{:>014.6a}'"), F(0.25));
  check(STR("answer is '1.000000p-2   '"), STR("answer is '{:<014.6a}'"), F(0.25));
  check(STR("answer is ' 1.000000p-2  '"), STR("answer is '{:^014.6a}'"), F(0.25));

  // *** Sign ***
  check(STR("answer is '0.000000p+0'"), STR("answer is '{:.6a}'"), F(0));
  check(STR("answer is '0.000000p+0'"), STR("answer is '{:-.6a}'"), F(0));
  check(STR("answer is '+0.000000p+0'"), STR("answer is '{:+.6a}'"), F(0));
  check(STR("answer is ' 0.000000p+0'"), STR("answer is '{: .6a}'"), F(0));

  check(STR("answer is '-0.000000p+0'"), STR("answer is '{:.6a}'"), F(-0.));
  check(STR("answer is '-0.000000p+0'"), STR("answer is '{:-.6a}'"), F(-0.));
  check(STR("answer is '-0.000000p+0'"), STR("answer is '{:+.6a}'"), F(-0.));
  check(STR("answer is '-0.000000p+0'"), STR("answer is '{: .6a}'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check(STR("answer is 'inf'"), STR("answer is '{:.6a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf'"), STR("answer is '{:-.6a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '+inf'"), STR("answer is '{:+.6a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is ' inf'"), STR("answer is '{: .6a}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '-inf'"), STR("answer is '{:.6a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:-.6a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:+.6a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{: .6a}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:.6a}'"), nan_pos);
  check(STR("answer is 'nan'"), STR("answer is '{:-.6a}'"), nan_pos);
  check(STR("answer is '+nan'"), STR("answer is '{:+.6a}'"), nan_pos);
  check(STR("answer is ' nan'"), STR("answer is '{: .6a}'"), nan_pos);

  check(STR("answer is '-nan'"), STR("answer is '{:.6a}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:-.6a}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:+.6a}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{: .6a}'"), nan_neg);

  // *** alternate form ***
  check(STR("answer is '1.400000p+1'"), STR("answer is '{:#.6a}'"), F(2.5));

  check(STR("answer is 'inf'"), STR("answer is '{:#.6a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:#.6a}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:#.6a}'"), nan_pos);
  check(STR("answer is '-nan'"), STR("answer is '{:#.6a}'"), nan_neg);

  // *** zero-padding & width ***
  check(STR("answer is '1.000000p-5'"), STR("answer is '{:011.6a}'"), 0.03125);
  check(STR("answer is '+1.000000p-5'"), STR("answer is '{:+012.6a}'"), 0.03125);
  check(STR("answer is '+01.000000p-5'"), STR("answer is '{:+013.6a}'"), 0.03125);

  check(STR("answer is '0001.000000p-5'"), STR("answer is '{:014.6a}'"), 0.03125);
  check(STR("answer is '0001.000000p-5'"), STR("answer is '{:-014.6a}'"), 0.03125);
  check(STR("answer is '+001.000000p-5'"), STR("answer is '{:+014.6a}'"), 0.03125);
  check(STR("answer is ' 001.000000p-5'"), STR("answer is '{: 014.6a}'"), 0.03125);

  check(STR("answer is '       inf'"), STR("answer is '{:010.6a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{:-010.6a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '      +inf'"), STR("answer is '{:+010.6a}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{: 010.6a}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '      -inf'"), STR("answer is '{:010.6a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:-010.6a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:+010.6a}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{: 010.6a}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '       nan'"), STR("answer is '{:010.6a}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{:-010.6a}'"), nan_pos);
  check(STR("answer is '      +nan'"), STR("answer is '{:+010.6a}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{: 010.6a}'"), nan_pos);

  check(STR("answer is '      -nan'"), STR("answer is '{:010.6a}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:-010.6a}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:+010.6a}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{: 010.6a}'"), nan_neg);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_hex_upper_case_precision(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN(); // "nan"
  auto nan_neg = std::copysign(nan_pos, -1.0);        // "-nan"

  // *** align-fill & width ***
  check(STR("answer is '   1.000000P-2'"), STR("answer is '{:14.6A}'"), F(0.25));
  check(STR("answer is '   1.000000P-2'"), STR("answer is '{:>14.6A}'"), F(0.25));
  check(STR("answer is '1.000000P-2   '"), STR("answer is '{:<14.6A}'"), F(0.25));
  check(STR("answer is ' 1.000000P-2  '"), STR("answer is '{:^14.6A}'"), F(0.25));

  check(STR("answer is '---1.000000P-3'"), STR("answer is '{:->14.6A}'"), F(125e-3));
  check(STR("answer is '1.000000P-3---'"), STR("answer is '{:-<14.6A}'"), F(125e-3));
  check(STR("answer is '-1.000000P-3--'"), STR("answer is '{:-^14.6A}'"), F(125e-3));

  check(STR("answer is '***INF'"), STR("answer is '{:*>6.6A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'INF***'"), STR("answer is '{:*<6.6A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '*INF**'"), STR("answer is '{:*^6.6A}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '###-INF'"), STR("answer is '{:#>7.6A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF###'"), STR("answer is '{:#<7.6A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '#-INF##'"), STR("answer is '{:#^7.6A}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '^^^NAN'"), STR("answer is '{:^>6.6A}'"), nan_pos);
  check(STR("answer is 'NAN^^^'"), STR("answer is '{:^<6.6A}'"), nan_pos);
  check(STR("answer is '^NAN^^'"), STR("answer is '{:^^6.6A}'"), nan_pos);

  check(STR("answer is '000-NAN'"), STR("answer is '{:0>7.6A}'"), nan_neg);
  check(STR("answer is '-NAN000'"), STR("answer is '{:0<7.6A}'"), nan_neg);
  check(STR("answer is '0-NAN00'"), STR("answer is '{:0^7.6A}'"), nan_neg);

  // Test whether zero padding is ignored
  check(STR("answer is '   1.000000P-2'"), STR("answer is '{:>014.6A}'"), F(0.25));
  check(STR("answer is '1.000000P-2   '"), STR("answer is '{:<014.6A}'"), F(0.25));
  check(STR("answer is ' 1.000000P-2  '"), STR("answer is '{:^014.6A}'"), F(0.25));

  // *** Sign ***
  check(STR("answer is '0.000000P+0'"), STR("answer is '{:.6A}'"), F(0));
  check(STR("answer is '0.000000P+0'"), STR("answer is '{:-.6A}'"), F(0));
  check(STR("answer is '+0.000000P+0'"), STR("answer is '{:+.6A}'"), F(0));
  check(STR("answer is ' 0.000000P+0'"), STR("answer is '{: .6A}'"), F(0));

  check(STR("answer is '-0.000000P+0'"), STR("answer is '{:.6A}'"), F(-0.));
  check(STR("answer is '-0.000000P+0'"), STR("answer is '{:-.6A}'"), F(-0.));
  check(STR("answer is '-0.000000P+0'"), STR("answer is '{:+.6A}'"), F(-0.));
  check(STR("answer is '-0.000000P+0'"), STR("answer is '{: .6A}'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check(STR("answer is 'INF'"), STR("answer is '{:.6A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'INF'"), STR("answer is '{:-.6A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '+INF'"), STR("answer is '{:+.6A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is ' INF'"), STR("answer is '{: .6A}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '-INF'"), STR("answer is '{:.6A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:-.6A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:+.6A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{: .6A}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'NAN'"), STR("answer is '{:.6A}'"), nan_pos);
  check(STR("answer is 'NAN'"), STR("answer is '{:-.6A}'"), nan_pos);
  check(STR("answer is '+NAN'"), STR("answer is '{:+.6A}'"), nan_pos);
  check(STR("answer is ' NAN'"), STR("answer is '{: .6A}'"), nan_pos);

  check(STR("answer is '-NAN'"), STR("answer is '{:.6A}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{:-.6A}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{:+.6A}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{: .6A}'"), nan_neg);

  // *** alternate form ***
  check(STR("answer is '1.400000P+1'"), STR("answer is '{:#.6A}'"), F(2.5));

  check(STR("answer is 'INF'"), STR("answer is '{:#.6A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:#.6A}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'NAN'"), STR("answer is '{:#.6A}'"), nan_pos);
  check(STR("answer is '-NAN'"), STR("answer is '{:#.6A}'"), nan_neg);

  // *** zero-padding & width ***
  check(STR("answer is '1.000000P-5'"), STR("answer is '{:011.6A}'"), 0.03125);
  check(STR("answer is '+1.000000P-5'"), STR("answer is '{:+012.6A}'"), 0.03125);
  check(STR("answer is '+01.000000P-5'"), STR("answer is '{:+013.6A}'"), 0.03125);

  check(STR("answer is '0001.000000P-5'"), STR("answer is '{:014.6A}'"), 0.03125);
  check(STR("answer is '0001.000000P-5'"), STR("answer is '{:-014.6A}'"), 0.03125);
  check(STR("answer is '+001.000000P-5'"), STR("answer is '{:+014.6A}'"), 0.03125);
  check(STR("answer is ' 001.000000P-5'"), STR("answer is '{: 014.6A}'"), 0.03125);

  check(STR("answer is '       INF'"), STR("answer is '{:010.6A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       INF'"), STR("answer is '{:-010.6A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '      +INF'"), STR("answer is '{:+010.6A}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       INF'"), STR("answer is '{: 010.6A}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '      -INF'"), STR("answer is '{:010.6A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{:-010.6A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{:+010.6A}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{: 010.6A}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '       NAN'"), STR("answer is '{:010.6A}'"), nan_pos);
  check(STR("answer is '       NAN'"), STR("answer is '{:-010.6A}'"), nan_pos);
  check(STR("answer is '      +NAN'"), STR("answer is '{:+010.6A}'"), nan_pos);
  check(STR("answer is '       NAN'"), STR("answer is '{: 010.6A}'"), nan_pos);

  check(STR("answer is '      -NAN'"), STR("answer is '{:010.6A}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{:-010.6A}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{:+010.6A}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{: 010.6A}'"), nan_neg);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_scientific_lower_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN(); // "nan"
  auto nan_neg = std::copysign(nan_pos, -1.0);        // "-nan"

  // *** align-fill & width ***
  check(STR("answer is '   2.500000e-01'"), STR("answer is '{:15e}'"), F(0.25));
  check(STR("answer is '   2.500000e-01'"), STR("answer is '{:>15e}'"), F(0.25));
  check(STR("answer is '2.500000e-01   '"), STR("answer is '{:<15e}'"), F(0.25));
  check(STR("answer is ' 2.500000e-01  '"), STR("answer is '{:^15e}'"), F(0.25));

  check(STR("answer is '---1.250000e-01'"), STR("answer is '{:->15e}'"), F(125e-3));
  check(STR("answer is '1.250000e-01---'"), STR("answer is '{:-<15e}'"), F(125e-3));
  check(STR("answer is '-1.250000e-01--'"), STR("answer is '{:-^15e}'"), F(125e-3));

  check(STR("answer is '***inf'"), STR("answer is '{:*>6e}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf***'"), STR("answer is '{:*<6e}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '*inf**'"), STR("answer is '{:*^6e}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '###-inf'"), STR("answer is '{:#>7e}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf###'"), STR("answer is '{:#<7e}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '#-inf##'"), STR("answer is '{:#^7e}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '^^^nan'"), STR("answer is '{:^>6e}'"), nan_pos);
  check(STR("answer is 'nan^^^'"), STR("answer is '{:^<6e}'"), nan_pos);
  check(STR("answer is '^nan^^'"), STR("answer is '{:^^6e}'"), nan_pos);

  check(STR("answer is '000-nan'"), STR("answer is '{:0>7e}'"), nan_neg);
  check(STR("answer is '-nan000'"), STR("answer is '{:0<7e}'"), nan_neg);
  check(STR("answer is '0-nan00'"), STR("answer is '{:0^7e}'"), nan_neg);

  // Test whether zero padding is ignored
  check(STR("answer is '   2.500000e-01'"), STR("answer is '{:>015e}'"), F(0.25));
  check(STR("answer is '2.500000e-01   '"), STR("answer is '{:<015e}'"), F(0.25));
  check(STR("answer is ' 2.500000e-01  '"), STR("answer is '{:^015e}'"), F(0.25));

  // *** Sign ***
  check(STR("answer is '0.000000e+00'"), STR("answer is '{:e}'"), F(0));
  check(STR("answer is '0.000000e+00'"), STR("answer is '{:-e}'"), F(0));
  check(STR("answer is '+0.000000e+00'"), STR("answer is '{:+e}'"), F(0));
  check(STR("answer is ' 0.000000e+00'"), STR("answer is '{: e}'"), F(0));

  check(STR("answer is '-0.000000e+00'"), STR("answer is '{:e}'"), F(-0.));
  check(STR("answer is '-0.000000e+00'"), STR("answer is '{:-e}'"), F(-0.));
  check(STR("answer is '-0.000000e+00'"), STR("answer is '{:+e}'"), F(-0.));
  check(STR("answer is '-0.000000e+00'"), STR("answer is '{: e}'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check(STR("answer is 'inf'"), STR("answer is '{:e}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf'"), STR("answer is '{:-e}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '+inf'"), STR("answer is '{:+e}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is ' inf'"), STR("answer is '{: e}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '-inf'"), STR("answer is '{:e}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:-e}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:+e}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{: e}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:e}'"), nan_pos);
  check(STR("answer is 'nan'"), STR("answer is '{:-e}'"), nan_pos);
  check(STR("answer is '+nan'"), STR("answer is '{:+e}'"), nan_pos);
  check(STR("answer is ' nan'"), STR("answer is '{: e}'"), nan_pos);

  check(STR("answer is '-nan'"), STR("answer is '{:e}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:-e}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:+e}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{: e}'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check(STR("answer is '0e+00'"), STR("answer is '{:.0e}'"), F(0));
  check(STR("answer is '0.e+00'"), STR("answer is '{:#.0e}'"), F(0));

  check(STR("answer is '0.000000e+00'"), STR("answer is '{:#e}'"), F(0));
  check(STR("answer is '2.500000e+00'"), STR("answer is '{:#e}'"), F(2.5));

  check(STR("answer is 'inf'"), STR("answer is '{:#e}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:#e}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:#e}'"), nan_pos);
  check(STR("answer is '-nan'"), STR("answer is '{:#e}'"), nan_neg);

  // *** zero-padding & width ***
  check(STR("answer is '3.125000e-02'"), STR("answer is '{:07e}'"), 0.03125);
  check(STR("answer is '+3.125000e-02'"), STR("answer is '{:+07e}'"), 0.03125);
  check(STR("answer is '+3.125000e-02'"), STR("answer is '{:+08e}'"), 0.03125);
  check(STR("answer is '+3.125000e-02'"), STR("answer is '{:+09e}'"), 0.03125);

  check(STR("answer is '003.125000e-02'"), STR("answer is '{:014e}'"), 0.03125);
  check(STR("answer is '003.125000e-02'"), STR("answer is '{:-014e}'"), 0.03125);
  check(STR("answer is '+03.125000e-02'"), STR("answer is '{:+014e}'"), 0.03125);
  check(STR("answer is ' 03.125000e-02'"), STR("answer is '{: 014e}'"), 0.03125);

  check(STR("answer is '       inf'"), STR("answer is '{:010e}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{:-010e}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '      +inf'"), STR("answer is '{:+010e}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{: 010e}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '      -inf'"), STR("answer is '{:010e}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:-010e}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:+010e}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{: 010e}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '       nan'"), STR("answer is '{:010e}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{:-010e}'"), nan_pos);
  check(STR("answer is '      +nan'"), STR("answer is '{:+010e}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{: 010e}'"), nan_pos);

  check(STR("answer is '      -nan'"), STR("answer is '{:010e}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:-010e}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:+010e}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{: 010e}'"), nan_neg);

  // *** precision ***
  check(STR("answer is '3e-02'"), STR("answer is '{:.0e}'"), 0.03125);
  check(STR("answer is '3.1e-02'"), STR("answer is '{:.1e}'"), 0.03125);
  check(STR("answer is '3.125e-02'"), STR("answer is '{:.3e}'"), 0.03125);
  check(STR("answer is '3.1250000000e-02'"), STR("answer is '{:.10e}'"), 0.03125);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_scientific_upper_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN(); // "nan"
  auto nan_neg = std::copysign(nan_pos, -1.0);        // "-nan"

  // *** align-fill & width ***
  check(STR("answer is '   2.500000E-01'"), STR("answer is '{:15E}'"), F(0.25));
  check(STR("answer is '   2.500000E-01'"), STR("answer is '{:>15E}'"), F(0.25));
  check(STR("answer is '2.500000E-01   '"), STR("answer is '{:<15E}'"), F(0.25));
  check(STR("answer is ' 2.500000E-01  '"), STR("answer is '{:^15E}'"), F(0.25));

  check(STR("answer is '---1.250000E-01'"), STR("answer is '{:->15E}'"), F(125e-3));
  check(STR("answer is '1.250000E-01---'"), STR("answer is '{:-<15E}'"), F(125e-3));
  check(STR("answer is '-1.250000E-01--'"), STR("answer is '{:-^15E}'"), F(125e-3));

  check(STR("answer is '***INF'"), STR("answer is '{:*>6E}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'INF***'"), STR("answer is '{:*<6E}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '*INF**'"), STR("answer is '{:*^6E}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '###-INF'"), STR("answer is '{:#>7E}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF###'"), STR("answer is '{:#<7E}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '#-INF##'"), STR("answer is '{:#^7E}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '^^^NAN'"), STR("answer is '{:^>6E}'"), nan_pos);
  check(STR("answer is 'NAN^^^'"), STR("answer is '{:^<6E}'"), nan_pos);
  check(STR("answer is '^NAN^^'"), STR("answer is '{:^^6E}'"), nan_pos);

  check(STR("answer is '000-NAN'"), STR("answer is '{:0>7E}'"), nan_neg);
  check(STR("answer is '-NAN000'"), STR("answer is '{:0<7E}'"), nan_neg);
  check(STR("answer is '0-NAN00'"), STR("answer is '{:0^7E}'"), nan_neg);

  // Test whether zero padding is ignored
  check(STR("answer is '   2.500000E-01'"), STR("answer is '{:>015E}'"), F(0.25));
  check(STR("answer is '2.500000E-01   '"), STR("answer is '{:<015E}'"), F(0.25));
  check(STR("answer is ' 2.500000E-01  '"), STR("answer is '{:^015E}'"), F(0.25));

  // *** Sign ***
  check(STR("answer is '0.000000E+00'"), STR("answer is '{:E}'"), F(0));
  check(STR("answer is '0.000000E+00'"), STR("answer is '{:-E}'"), F(0));
  check(STR("answer is '+0.000000E+00'"), STR("answer is '{:+E}'"), F(0));
  check(STR("answer is ' 0.000000E+00'"), STR("answer is '{: E}'"), F(0));

  check(STR("answer is '-0.000000E+00'"), STR("answer is '{:E}'"), F(-0.));
  check(STR("answer is '-0.000000E+00'"), STR("answer is '{:-E}'"), F(-0.));
  check(STR("answer is '-0.000000E+00'"), STR("answer is '{:+E}'"), F(-0.));
  check(STR("answer is '-0.000000E+00'"), STR("answer is '{: E}'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check(STR("answer is 'INF'"), STR("answer is '{:E}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'INF'"), STR("answer is '{:-E}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '+INF'"), STR("answer is '{:+E}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is ' INF'"), STR("answer is '{: E}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '-INF'"), STR("answer is '{:E}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:-E}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:+E}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{: E}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'NAN'"), STR("answer is '{:E}'"), nan_pos);
  check(STR("answer is 'NAN'"), STR("answer is '{:-E}'"), nan_pos);
  check(STR("answer is '+NAN'"), STR("answer is '{:+E}'"), nan_pos);
  check(STR("answer is ' NAN'"), STR("answer is '{: E}'"), nan_pos);

  check(STR("answer is '-NAN'"), STR("answer is '{:E}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{:-E}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{:+E}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{: E}'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check(STR("answer is '0E+00'"), STR("answer is '{:.0E}'"), F(0));
  check(STR("answer is '0.E+00'"), STR("answer is '{:#.0E}'"), F(0));

  check(STR("answer is '0.000000E+00'"), STR("answer is '{:#E}'"), F(0));
  check(STR("answer is '2.500000E+00'"), STR("answer is '{:#E}'"), F(2.5));

  check(STR("answer is 'INF'"), STR("answer is '{:#E}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:#E}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'NAN'"), STR("answer is '{:#E}'"), nan_pos);
  check(STR("answer is '-NAN'"), STR("answer is '{:#E}'"), nan_neg);

  // *** zero-padding & width ***
  check(STR("answer is '3.125000E-02'"), STR("answer is '{:07E}'"), 0.03125);
  check(STR("answer is '+3.125000E-02'"), STR("answer is '{:+07E}'"), 0.03125);
  check(STR("answer is '+3.125000E-02'"), STR("answer is '{:+08E}'"), 0.03125);
  check(STR("answer is '+3.125000E-02'"), STR("answer is '{:+09E}'"), 0.03125);

  check(STR("answer is '003.125000E-02'"), STR("answer is '{:014E}'"), 0.03125);
  check(STR("answer is '003.125000E-02'"), STR("answer is '{:-014E}'"), 0.03125);
  check(STR("answer is '+03.125000E-02'"), STR("answer is '{:+014E}'"), 0.03125);
  check(STR("answer is ' 03.125000E-02'"), STR("answer is '{: 014E}'"), 0.03125);

  check(STR("answer is '       INF'"), STR("answer is '{:010E}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       INF'"), STR("answer is '{:-010E}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '      +INF'"), STR("answer is '{:+010E}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       INF'"), STR("answer is '{: 010E}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '      -INF'"), STR("answer is '{:010E}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{:-010E}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{:+010E}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{: 010E}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '       NAN'"), STR("answer is '{:010E}'"), nan_pos);
  check(STR("answer is '       NAN'"), STR("answer is '{:-010E}'"), nan_pos);
  check(STR("answer is '      +NAN'"), STR("answer is '{:+010E}'"), nan_pos);
  check(STR("answer is '       NAN'"), STR("answer is '{: 010E}'"), nan_pos);

  check(STR("answer is '      -NAN'"), STR("answer is '{:010E}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{:-010E}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{:+010E}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{: 010E}'"), nan_neg);

  // *** precision ***
  check(STR("answer is '3E-02'"), STR("answer is '{:.0E}'"), 0.03125);
  check(STR("answer is '3.1E-02'"), STR("answer is '{:.1E}'"), 0.03125);
  check(STR("answer is '3.125E-02'"), STR("answer is '{:.3E}'"), 0.03125);
  check(STR("answer is '3.1250000000E-02'"), STR("answer is '{:.10E}'"), 0.03125);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_fixed_lower_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN(); // "nan"
  auto nan_neg = std::copysign(nan_pos, -1.0);        // "-nan"

  // *** align-fill & width ***
  check(STR("answer is '   0.250000'"), STR("answer is '{:11f}'"), F(0.25));
  check(STR("answer is '   0.250000'"), STR("answer is '{:>11f}'"), F(0.25));
  check(STR("answer is '0.250000   '"), STR("answer is '{:<11f}'"), F(0.25));
  check(STR("answer is ' 0.250000  '"), STR("answer is '{:^11f}'"), F(0.25));

  check(STR("answer is '---0.125000'"), STR("answer is '{:->11f}'"), F(125e-3));
  check(STR("answer is '0.125000---'"), STR("answer is '{:-<11f}'"), F(125e-3));
  check(STR("answer is '-0.125000--'"), STR("answer is '{:-^11f}'"), F(125e-3));

  check(STR("answer is '***inf'"), STR("answer is '{:*>6f}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf***'"), STR("answer is '{:*<6f}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '*inf**'"), STR("answer is '{:*^6f}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '###-inf'"), STR("answer is '{:#>7f}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf###'"), STR("answer is '{:#<7f}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '#-inf##'"), STR("answer is '{:#^7f}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '^^^nan'"), STR("answer is '{:^>6f}'"), nan_pos);
  check(STR("answer is 'nan^^^'"), STR("answer is '{:^<6f}'"), nan_pos);
  check(STR("answer is '^nan^^'"), STR("answer is '{:^^6f}'"), nan_pos);

  check(STR("answer is '000-nan'"), STR("answer is '{:0>7f}'"), nan_neg);
  check(STR("answer is '-nan000'"), STR("answer is '{:0<7f}'"), nan_neg);
  check(STR("answer is '0-nan00'"), STR("answer is '{:0^7f}'"), nan_neg);

  // Test whether zero padding is ignored
  check(STR("answer is '   0.250000'"), STR("answer is '{:>011f}'"), F(0.25));
  check(STR("answer is '0.250000   '"), STR("answer is '{:<011f}'"), F(0.25));
  check(STR("answer is ' 0.250000  '"), STR("answer is '{:^011f}'"), F(0.25));

  // *** Sign ***
  check(STR("answer is '0.000000'"), STR("answer is '{:f}'"), F(0));
  check(STR("answer is '0.000000'"), STR("answer is '{:-f}'"), F(0));
  check(STR("answer is '+0.000000'"), STR("answer is '{:+f}'"), F(0));
  check(STR("answer is ' 0.000000'"), STR("answer is '{: f}'"), F(0));

  check(STR("answer is '-0.000000'"), STR("answer is '{:f}'"), F(-0.));
  check(STR("answer is '-0.000000'"), STR("answer is '{:-f}'"), F(-0.));
  check(STR("answer is '-0.000000'"), STR("answer is '{:+f}'"), F(-0.));
  check(STR("answer is '-0.000000'"), STR("answer is '{: f}'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check(STR("answer is 'inf'"), STR("answer is '{:f}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf'"), STR("answer is '{:-f}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '+inf'"), STR("answer is '{:+f}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is ' inf'"), STR("answer is '{: f}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '-inf'"), STR("answer is '{:f}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:-f}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:+f}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{: f}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:f}'"), nan_pos);
  check(STR("answer is 'nan'"), STR("answer is '{:-f}'"), nan_pos);
  check(STR("answer is '+nan'"), STR("answer is '{:+f}'"), nan_pos);
  check(STR("answer is ' nan'"), STR("answer is '{: f}'"), nan_pos);

  check(STR("answer is '-nan'"), STR("answer is '{:f}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:-f}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:+f}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{: f}'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check(STR("answer is '0'"), STR("answer is '{:.0f}'"), F(0));
  check(STR("answer is '0.'"), STR("answer is '{:#.0f}'"), F(0));

  check(STR("answer is '0.000000'"), STR("answer is '{:#f}'"), F(0));
  check(STR("answer is '2.500000'"), STR("answer is '{:#f}'"), F(2.5));

  check(STR("answer is 'inf'"), STR("answer is '{:#f}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:#f}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:#f}'"), nan_pos);
  check(STR("answer is '-nan'"), STR("answer is '{:#f}'"), nan_neg);

  // *** zero-padding & width ***
  check(STR("answer is '0.031250'"), STR("answer is '{:07f}'"), 0.03125);
  check(STR("answer is '+0.031250'"), STR("answer is '{:+07f}'"), 0.03125);
  check(STR("answer is '+0.031250'"), STR("answer is '{:+08f}'"), 0.03125);
  check(STR("answer is '+0.031250'"), STR("answer is '{:+09f}'"), 0.03125);

  check(STR("answer is '000.031250'"), STR("answer is '{:010f}'"), 0.03125);
  check(STR("answer is '000.031250'"), STR("answer is '{:-010f}'"), 0.03125);
  check(STR("answer is '+00.031250'"), STR("answer is '{:+010f}'"), 0.03125);
  check(STR("answer is ' 00.031250'"), STR("answer is '{: 010f}'"), 0.03125);

  check(STR("answer is '       inf'"), STR("answer is '{:010f}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{:-010f}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '      +inf'"), STR("answer is '{:+010f}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{: 010f}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '      -inf'"), STR("answer is '{:010f}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:-010f}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:+010f}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{: 010f}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '       nan'"), STR("answer is '{:010f}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{:-010f}'"), nan_pos);
  check(STR("answer is '      +nan'"), STR("answer is '{:+010f}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{: 010f}'"), nan_pos);

  check(STR("answer is '      -nan'"), STR("answer is '{:010f}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:-010f}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:+010f}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{: 010f}'"), nan_neg);

  // *** precision ***
  check(STR("answer is '0'"), STR("answer is '{:.0f}'"), 0.03125);
  check(STR("answer is '0.0'"), STR("answer is '{:.1f}'"), 0.03125);
  check(STR("answer is '0.03125'"), STR("answer is '{:.5f}'"), 0.03125);
  check(STR("answer is '0.0312500000'"), STR("answer is '{:.10f}'"), 0.03125);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_fixed_upper_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN(); // "nan"
  auto nan_neg = std::copysign(nan_pos, -1.0);        // "-nan"

  // *** align-fill & width ***
  check(STR("answer is '   0.250000'"), STR("answer is '{:11F}'"), F(0.25));
  check(STR("answer is '   0.250000'"), STR("answer is '{:>11F}'"), F(0.25));
  check(STR("answer is '0.250000   '"), STR("answer is '{:<11F}'"), F(0.25));
  check(STR("answer is ' 0.250000  '"), STR("answer is '{:^11F}'"), F(0.25));

  check(STR("answer is '---0.125000'"), STR("answer is '{:->11F}'"), F(125e-3));
  check(STR("answer is '0.125000---'"), STR("answer is '{:-<11F}'"), F(125e-3));
  check(STR("answer is '-0.125000--'"), STR("answer is '{:-^11F}'"), F(125e-3));

  check(STR("answer is '***INF'"), STR("answer is '{:*>6F}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'INF***'"), STR("answer is '{:*<6F}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '*INF**'"), STR("answer is '{:*^6F}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '###-INF'"), STR("answer is '{:#>7F}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF###'"), STR("answer is '{:#<7F}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '#-INF##'"), STR("answer is '{:#^7F}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '^^^NAN'"), STR("answer is '{:^>6F}'"), nan_pos);
  check(STR("answer is 'NAN^^^'"), STR("answer is '{:^<6F}'"), nan_pos);
  check(STR("answer is '^NAN^^'"), STR("answer is '{:^^6F}'"), nan_pos);

  check(STR("answer is '000-NAN'"), STR("answer is '{:0>7F}'"), nan_neg);
  check(STR("answer is '-NAN000'"), STR("answer is '{:0<7F}'"), nan_neg);
  check(STR("answer is '0-NAN00'"), STR("answer is '{:0^7F}'"), nan_neg);

  // Test whether zero padding is ignored
  check(STR("answer is '   0.250000'"), STR("answer is '{:>011F}'"), F(0.25));
  check(STR("answer is '0.250000   '"), STR("answer is '{:<011F}'"), F(0.25));
  check(STR("answer is ' 0.250000  '"), STR("answer is '{:^011F}'"), F(0.25));

  // *** Sign ***
  check(STR("answer is '0.000000'"), STR("answer is '{:F}'"), F(0));
  check(STR("answer is '0.000000'"), STR("answer is '{:-F}'"), F(0));
  check(STR("answer is '+0.000000'"), STR("answer is '{:+F}'"), F(0));
  check(STR("answer is ' 0.000000'"), STR("answer is '{: F}'"), F(0));

  check(STR("answer is '-0.000000'"), STR("answer is '{:F}'"), F(-0.));
  check(STR("answer is '-0.000000'"), STR("answer is '{:-F}'"), F(-0.));
  check(STR("answer is '-0.000000'"), STR("answer is '{:+F}'"), F(-0.));
  check(STR("answer is '-0.000000'"), STR("answer is '{: F}'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check(STR("answer is 'INF'"), STR("answer is '{:F}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'INF'"), STR("answer is '{:-F}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '+INF'"), STR("answer is '{:+F}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is ' INF'"), STR("answer is '{: F}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '-INF'"), STR("answer is '{:F}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:-F}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:+F}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{: F}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'NAN'"), STR("answer is '{:F}'"), nan_pos);
  check(STR("answer is 'NAN'"), STR("answer is '{:-F}'"), nan_pos);
  check(STR("answer is '+NAN'"), STR("answer is '{:+F}'"), nan_pos);
  check(STR("answer is ' NAN'"), STR("answer is '{: F}'"), nan_pos);

  check(STR("answer is '-NAN'"), STR("answer is '{:F}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{:-F}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{:+F}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{: F}'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check(STR("answer is '0'"), STR("answer is '{:.0F}'"), F(0));
  check(STR("answer is '0.'"), STR("answer is '{:#.0F}'"), F(0));

  check(STR("answer is '0.000000'"), STR("answer is '{:#F}'"), F(0));
  check(STR("answer is '2.500000'"), STR("answer is '{:#F}'"), F(2.5));

  check(STR("answer is 'INF'"), STR("answer is '{:#F}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:#F}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'NAN'"), STR("answer is '{:#F}'"), nan_pos);
  check(STR("answer is '-NAN'"), STR("answer is '{:#F}'"), nan_neg);

  // *** zero-padding & width ***
  check(STR("answer is '0.031250'"), STR("answer is '{:07F}'"), 0.03125);
  check(STR("answer is '+0.031250'"), STR("answer is '{:+07F}'"), 0.03125);
  check(STR("answer is '+0.031250'"), STR("answer is '{:+08F}'"), 0.03125);
  check(STR("answer is '+0.031250'"), STR("answer is '{:+09F}'"), 0.03125);

  check(STR("answer is '000.031250'"), STR("answer is '{:010F}'"), 0.03125);
  check(STR("answer is '000.031250'"), STR("answer is '{:-010F}'"), 0.03125);
  check(STR("answer is '+00.031250'"), STR("answer is '{:+010F}'"), 0.03125);
  check(STR("answer is ' 00.031250'"), STR("answer is '{: 010F}'"), 0.03125);

  check(STR("answer is '       INF'"), STR("answer is '{:010F}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       INF'"), STR("answer is '{:-010F}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '      +INF'"), STR("answer is '{:+010F}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       INF'"), STR("answer is '{: 010F}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '      -INF'"), STR("answer is '{:010F}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{:-010F}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{:+010F}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{: 010F}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '       NAN'"), STR("answer is '{:010F}'"), nan_pos);
  check(STR("answer is '       NAN'"), STR("answer is '{:-010F}'"), nan_pos);
  check(STR("answer is '      +NAN'"), STR("answer is '{:+010F}'"), nan_pos);
  check(STR("answer is '       NAN'"), STR("answer is '{: 010F}'"), nan_pos);

  check(STR("answer is '      -NAN'"), STR("answer is '{:010F}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{:-010F}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{:+010F}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{: 010F}'"), nan_neg);

  // *** precision ***
  check(STR("answer is '0'"), STR("answer is '{:.0F}'"), 0.03125);
  check(STR("answer is '0.0'"), STR("answer is '{:.1F}'"), 0.03125);
  check(STR("answer is '0.03125'"), STR("answer is '{:.5F}'"), 0.03125);
  check(STR("answer is '0.0312500000'"), STR("answer is '{:.10F}'"), 0.03125);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_general_lower_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN(); // "nan"
  auto nan_neg = std::copysign(nan_pos, -1.0);        // "-nan"

  // *** align-fill & width ***
  check(STR("answer is '   0.25'"), STR("answer is '{:7g}'"), F(0.25));
  check(STR("answer is '   0.25'"), STR("answer is '{:>7g}'"), F(0.25));
  check(STR("answer is '0.25   '"), STR("answer is '{:<7g}'"), F(0.25));
  check(STR("answer is ' 0.25  '"), STR("answer is '{:^7g}'"), F(0.25));

  check(STR("answer is '---0.125'"), STR("answer is '{:->8g}'"), F(125e-3));
  check(STR("answer is '0.125---'"), STR("answer is '{:-<8g}'"), F(125e-3));
  check(STR("answer is '-0.125--'"), STR("answer is '{:-^8g}'"), F(125e-3));

  check(STR("answer is '***inf'"), STR("answer is '{:*>6g}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf***'"), STR("answer is '{:*<6g}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '*inf**'"), STR("answer is '{:*^6g}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '###-inf'"), STR("answer is '{:#>7g}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf###'"), STR("answer is '{:#<7g}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '#-inf##'"), STR("answer is '{:#^7g}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '^^^nan'"), STR("answer is '{:^>6g}'"), nan_pos);
  check(STR("answer is 'nan^^^'"), STR("answer is '{:^<6g}'"), nan_pos);
  check(STR("answer is '^nan^^'"), STR("answer is '{:^^6g}'"), nan_pos);

  check(STR("answer is '000-nan'"), STR("answer is '{:0>7g}'"), nan_neg);
  check(STR("answer is '-nan000'"), STR("answer is '{:0<7g}'"), nan_neg);
  check(STR("answer is '0-nan00'"), STR("answer is '{:0^7g}'"), nan_neg);

  // Test whether zero padding is ignored
  check(STR("answer is '   0.25'"), STR("answer is '{:>07g}'"), F(0.25));
  check(STR("answer is '0.25   '"), STR("answer is '{:<07g}'"), F(0.25));
  check(STR("answer is ' 0.25  '"), STR("answer is '{:^07g}'"), F(0.25));

  // *** Sign ***
  check(STR("answer is '0'"), STR("answer is '{:g}'"), F(0));
  check(STR("answer is '0'"), STR("answer is '{:-g}'"), F(0));
  check(STR("answer is '+0'"), STR("answer is '{:+g}'"), F(0));
  check(STR("answer is ' 0'"), STR("answer is '{: g}'"), F(0));

  check(STR("answer is '-0'"), STR("answer is '{:g}'"), F(-0.));
  check(STR("answer is '-0'"), STR("answer is '{:-g}'"), F(-0.));
  check(STR("answer is '-0'"), STR("answer is '{:+g}'"), F(-0.));
  check(STR("answer is '-0'"), STR("answer is '{: g}'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check(STR("answer is 'inf'"), STR("answer is '{:g}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf'"), STR("answer is '{:-g}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '+inf'"), STR("answer is '{:+g}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is ' inf'"), STR("answer is '{: g}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '-inf'"), STR("answer is '{:g}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:-g}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:+g}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{: g}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:g}'"), nan_pos);
  check(STR("answer is 'nan'"), STR("answer is '{:-g}'"), nan_pos);
  check(STR("answer is '+nan'"), STR("answer is '{:+g}'"), nan_pos);
  check(STR("answer is ' nan'"), STR("answer is '{: g}'"), nan_pos);

  check(STR("answer is '-nan'"), STR("answer is '{:g}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:-g}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:+g}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{: g}'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check(STR("answer is '0'"), STR("answer is '{:.0g}'"), F(0));
  check(STR("answer is '0.'"), STR("answer is '{:#.0g}'"), F(0));

  check(STR("answer is '0.'"), STR("answer is '{:#g}'"), F(0));
  check(STR("answer is '2.5'"), STR("answer is '{:#g}'"), F(2.5));

  check(STR("answer is 'inf'"), STR("answer is '{:#g}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:#g}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:#g}'"), nan_pos);
  check(STR("answer is '-nan'"), STR("answer is '{:#g}'"), nan_neg);

  // *** zero-padding & width ***
  check(STR("answer is '0.03125'"), STR("answer is '{:06g}'"), 0.03125);
  check(STR("answer is '+0.03125'"), STR("answer is '{:+06g}'"), 0.03125);
  check(STR("answer is '+0.03125'"), STR("answer is '{:+07g}'"), 0.03125);
  check(STR("answer is '+0.03125'"), STR("answer is '{:+08g}'"), 0.03125);

  check(STR("answer is '000.03125'"), STR("answer is '{:09g}'"), 0.03125);
  check(STR("answer is '000.03125'"), STR("answer is '{:-09g}'"), 0.03125);
  check(STR("answer is '+00.03125'"), STR("answer is '{:+09g}'"), 0.03125);
  check(STR("answer is ' 00.03125'"), STR("answer is '{: 09g}'"), 0.03125);

  check(STR("answer is '       inf'"), STR("answer is '{:010g}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{:-010g}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '      +inf'"), STR("answer is '{:+010g}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{: 010g}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '      -inf'"), STR("answer is '{:010g}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:-010g}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:+010g}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{: 010g}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '       nan'"), STR("answer is '{:010g}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{:-010g}'"), nan_pos);
  check(STR("answer is '      +nan'"), STR("answer is '{:+010g}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{: 010g}'"), nan_pos);

  check(STR("answer is '      -nan'"), STR("answer is '{:010g}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:-010g}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:+010g}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{: 010g}'"), nan_neg);

  // *** precision ***
  check(STR("answer is '0.03'"), STR("answer is '{:.0g}'"), 0.03125);
  check(STR("answer is '0.03'"), STR("answer is '{:.1g}'"), 0.03125);
  check(STR("answer is '0.031'"), STR("answer is '{:.2g}'"), 0.03125);
  check(STR("answer is '0.0312'"), STR("answer is '{:.3g}'"), 0.03125);
  check(STR("answer is '0.03125'"), STR("answer is '{:.4g}'"), 0.03125);
  check(STR("answer is '0.03125'"), STR("answer is '{:.5g}'"), 0.03125);
  check(STR("answer is '0.03125'"), STR("answer is '{:.10g}'"), 0.03125);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_general_upper_case(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN(); // "nan"
  auto nan_neg = std::copysign(nan_pos, -1.0);        // "-nan"

  // *** align-fill & width ***
  check(STR("answer is '   0.25'"), STR("answer is '{:7G}'"), F(0.25));
  check(STR("answer is '   0.25'"), STR("answer is '{:>7G}'"), F(0.25));
  check(STR("answer is '0.25   '"), STR("answer is '{:<7G}'"), F(0.25));
  check(STR("answer is ' 0.25  '"), STR("answer is '{:^7G}'"), F(0.25));

  check(STR("answer is '---0.125'"), STR("answer is '{:->8G}'"), F(125e-3));
  check(STR("answer is '0.125---'"), STR("answer is '{:-<8G}'"), F(125e-3));
  check(STR("answer is '-0.125--'"), STR("answer is '{:-^8G}'"), F(125e-3));

  check(STR("answer is '***INF'"), STR("answer is '{:*>6G}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'INF***'"), STR("answer is '{:*<6G}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '*INF**'"), STR("answer is '{:*^6G}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '###-INF'"), STR("answer is '{:#>7G}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF###'"), STR("answer is '{:#<7G}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '#-INF##'"), STR("answer is '{:#^7G}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '^^^NAN'"), STR("answer is '{:^>6G}'"), nan_pos);
  check(STR("answer is 'NAN^^^'"), STR("answer is '{:^<6G}'"), nan_pos);
  check(STR("answer is '^NAN^^'"), STR("answer is '{:^^6G}'"), nan_pos);

  check(STR("answer is '000-NAN'"), STR("answer is '{:0>7G}'"), nan_neg);
  check(STR("answer is '-NAN000'"), STR("answer is '{:0<7G}'"), nan_neg);
  check(STR("answer is '0-NAN00'"), STR("answer is '{:0^7G}'"), nan_neg);

  // Test whether zero padding is ignored
  check(STR("answer is '   0.25'"), STR("answer is '{:>07G}'"), F(0.25));
  check(STR("answer is '0.25   '"), STR("answer is '{:<07G}'"), F(0.25));
  check(STR("answer is ' 0.25  '"), STR("answer is '{:^07G}'"), F(0.25));

  // *** Sign ***
  check(STR("answer is '0'"), STR("answer is '{:G}'"), F(0));
  check(STR("answer is '0'"), STR("answer is '{:-G}'"), F(0));
  check(STR("answer is '+0'"), STR("answer is '{:+G}'"), F(0));
  check(STR("answer is ' 0'"), STR("answer is '{: G}'"), F(0));

  check(STR("answer is '-0'"), STR("answer is '{:G}'"), F(-0.));
  check(STR("answer is '-0'"), STR("answer is '{:-G}'"), F(-0.));
  check(STR("answer is '-0'"), STR("answer is '{:+G}'"), F(-0.));
  check(STR("answer is '-0'"), STR("answer is '{: G}'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check(STR("answer is 'INF'"), STR("answer is '{:G}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'INF'"), STR("answer is '{:-G}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '+INF'"), STR("answer is '{:+G}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is ' INF'"), STR("answer is '{: G}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '-INF'"), STR("answer is '{:G}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:-G}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:+G}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{: G}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'NAN'"), STR("answer is '{:G}'"), nan_pos);
  check(STR("answer is 'NAN'"), STR("answer is '{:-G}'"), nan_pos);
  check(STR("answer is '+NAN'"), STR("answer is '{:+G}'"), nan_pos);
  check(STR("answer is ' NAN'"), STR("answer is '{: G}'"), nan_pos);

  check(STR("answer is '-NAN'"), STR("answer is '{:G}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{:-G}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{:+G}'"), nan_neg);
  check(STR("answer is '-NAN'"), STR("answer is '{: G}'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check(STR("answer is '0'"), STR("answer is '{:.0G}'"), F(0));
  check(STR("answer is '0.'"), STR("answer is '{:#.0G}'"), F(0));

  check(STR("answer is '0.'"), STR("answer is '{:#G}'"), F(0));
  check(STR("answer is '2.5'"), STR("answer is '{:#G}'"), F(2.5));

  check(STR("answer is 'INF'"), STR("answer is '{:#G}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '-INF'"), STR("answer is '{:#G}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'NAN'"), STR("answer is '{:#G}'"), nan_pos);
  check(STR("answer is '-NAN'"), STR("answer is '{:#G}'"), nan_neg);

  // *** zero-padding & width ***
  check(STR("answer is '0.03125'"), STR("answer is '{:06G}'"), 0.03125);
  check(STR("answer is '+0.03125'"), STR("answer is '{:+06G}'"), 0.03125);
  check(STR("answer is '+0.03125'"), STR("answer is '{:+07G}'"), 0.03125);
  check(STR("answer is '+0.03125'"), STR("answer is '{:+08G}'"), 0.03125);

  check(STR("answer is '000.03125'"), STR("answer is '{:09G}'"), 0.03125);
  check(STR("answer is '000.03125'"), STR("answer is '{:-09G}'"), 0.03125);
  check(STR("answer is '+00.03125'"), STR("answer is '{:+09G}'"), 0.03125);
  check(STR("answer is ' 00.03125'"), STR("answer is '{: 09G}'"), 0.03125);

  check(STR("answer is '       INF'"), STR("answer is '{:010G}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       INF'"), STR("answer is '{:-010G}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '      +INF'"), STR("answer is '{:+010G}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       INF'"), STR("answer is '{: 010G}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '      -INF'"), STR("answer is '{:010G}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{:-010G}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{:+010G}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -INF'"), STR("answer is '{: 010G}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '       NAN'"), STR("answer is '{:010G}'"), nan_pos);
  check(STR("answer is '       NAN'"), STR("answer is '{:-010G}'"), nan_pos);
  check(STR("answer is '      +NAN'"), STR("answer is '{:+010G}'"), nan_pos);
  check(STR("answer is '       NAN'"), STR("answer is '{: 010G}'"), nan_pos);

  check(STR("answer is '      -NAN'"), STR("answer is '{:010G}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{:-010G}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{:+010G}'"), nan_neg);
  check(STR("answer is '      -NAN'"), STR("answer is '{: 010G}'"), nan_neg);

  // *** precision ***
  check(STR("answer is '0.03'"), STR("answer is '{:.0G}'"), 0.03125);
  check(STR("answer is '0.03'"), STR("answer is '{:.1G}'"), 0.03125);
  check(STR("answer is '0.031'"), STR("answer is '{:.2G}'"), 0.03125);
  check(STR("answer is '0.0312'"), STR("answer is '{:.3G}'"), 0.03125);
  check(STR("answer is '0.03125'"), STR("answer is '{:.4G}'"), 0.03125);
  check(STR("answer is '0.03125'"), STR("answer is '{:.5G}'"), 0.03125);
  check(STR("answer is '0.03125'"), STR("answer is '{:.10G}'"), 0.03125);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_default(TestFunction check) {
  auto nan_pos = std::numeric_limits<F>::quiet_NaN(); // "nan"
  auto nan_neg = std::copysign(nan_pos, -1.0);        // "-nan"

  // *** align-fill & width ***
  check(STR("answer is '   0.25'"), STR("answer is '{:7}'"), F(0.25));
  check(STR("answer is '   0.25'"), STR("answer is '{:>7}'"), F(0.25));
  check(STR("answer is '0.25   '"), STR("answer is '{:<7}'"), F(0.25));
  check(STR("answer is ' 0.25  '"), STR("answer is '{:^7}'"), F(0.25));

  check(STR("answer is '---0.125'"), STR("answer is '{:->8}'"), F(125e-3));
  check(STR("answer is '0.125---'"), STR("answer is '{:-<8}'"), F(125e-3));
  check(STR("answer is '-0.125--'"), STR("answer is '{:-^8}'"), F(125e-3));

  check(STR("answer is '***inf'"), STR("answer is '{:*>6}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf***'"), STR("answer is '{:*<6}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '*inf**'"), STR("answer is '{:*^6}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '###-inf'"), STR("answer is '{:#>7}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf###'"), STR("answer is '{:#<7}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '#-inf##'"), STR("answer is '{:#^7}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '^^^nan'"), STR("answer is '{:^>6}'"), nan_pos);
  check(STR("answer is 'nan^^^'"), STR("answer is '{:^<6}'"), nan_pos);
  check(STR("answer is '^nan^^'"), STR("answer is '{:^^6}'"), nan_pos);

  check(STR("answer is '000-nan'"), STR("answer is '{:0>7}'"), nan_neg);
  check(STR("answer is '-nan000'"), STR("answer is '{:0<7}'"), nan_neg);
  check(STR("answer is '0-nan00'"), STR("answer is '{:0^7}'"), nan_neg);

  // Test whether zero padding is ignored
  check(STR("answer is '   0.25'"), STR("answer is '{:>07}'"), F(0.25));
  check(STR("answer is '0.25   '"), STR("answer is '{:<07}'"), F(0.25));
  check(STR("answer is ' 0.25  '"), STR("answer is '{:^07}'"), F(0.25));

  // *** Sign ***
  check(STR("answer is '0'"), STR("answer is '{:}'"), F(0));
  check(STR("answer is '0'"), STR("answer is '{:-}'"), F(0));
  check(STR("answer is '+0'"), STR("answer is '{:+}'"), F(0));
  check(STR("answer is ' 0'"), STR("answer is '{: }'"), F(0));

  check(STR("answer is '-0'"), STR("answer is '{:}'"), F(-0.));
  check(STR("answer is '-0'"), STR("answer is '{:-}'"), F(-0.));
  check(STR("answer is '-0'"), STR("answer is '{:+}'"), F(-0.));
  check(STR("answer is '-0'"), STR("answer is '{: }'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check(STR("answer is 'inf'"), STR("answer is '{:}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf'"), STR("answer is '{:-}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '+inf'"), STR("answer is '{:+}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is ' inf'"), STR("answer is '{: }'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '-inf'"), STR("answer is '{:}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:-}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:+}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{: }'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:}'"), nan_pos);
  check(STR("answer is 'nan'"), STR("answer is '{:-}'"), nan_pos);
  check(STR("answer is '+nan'"), STR("answer is '{:+}'"), nan_pos);
  check(STR("answer is ' nan'"), STR("answer is '{: }'"), nan_pos);

  check(STR("answer is '-nan'"), STR("answer is '{:}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:-}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:+}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{: }'"), nan_neg);

  // *** alternate form ***
  check(STR("answer is '0.'"), STR("answer is '{:#}'"), F(0));
  check(STR("answer is '2.5'"), STR("answer is '{:#}'"), F(2.5));

  check(STR("answer is 'inf'"), STR("answer is '{:#}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:#}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:#}'"), nan_pos);
  check(STR("answer is '-nan'"), STR("answer is '{:#}'"), nan_neg);

  // *** zero-padding & width ***
  check(STR("answer is '0.03125'"), STR("answer is '{:07}'"), 0.03125);
  check(STR("answer is '+0.03125'"), STR("answer is '{:+07}'"), 0.03125);
  check(STR("answer is '+0.03125'"), STR("answer is '{:+08}'"), 0.03125);
  check(STR("answer is '+00.03125'"), STR("answer is '{:+09}'"), 0.03125);

  check(STR("answer is '0000.03125'"), STR("answer is '{:010}'"), 0.03125);
  check(STR("answer is '0000.03125'"), STR("answer is '{:-010}'"), 0.03125);
  check(STR("answer is '+000.03125'"), STR("answer is '{:+010}'"), 0.03125);
  check(STR("answer is ' 000.03125'"), STR("answer is '{: 010}'"), 0.03125);

  check(STR("answer is '       inf'"), STR("answer is '{:010}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{:-010}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '      +inf'"), STR("answer is '{:+010}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{: 010}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '      -inf'"), STR("answer is '{:010}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:-010}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:+010}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{: 010}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '       nan'"), STR("answer is '{:010}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{:-010}'"), nan_pos);
  check(STR("answer is '      +nan'"), STR("answer is '{:+010}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{: 010}'"), nan_pos);

  check(STR("answer is '      -nan'"), STR("answer is '{:010}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:-010}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:+010}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{: 010}'"), nan_neg);

  // *** precision ***
  // See format_test_floating_point_default_precision

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp
}

template <class F, class CharT, class TestFunction>
void format_test_floating_point_default_precision(TestFunction check) {

  auto nan_pos = std::numeric_limits<F>::quiet_NaN(); // "nan"
  auto nan_neg = std::copysign(nan_pos, -1.0);        // "-nan"

  // *** align-fill & width ***
  check(STR("answer is '   0.25'"), STR("answer is '{:7.6}'"), F(0.25));
  check(STR("answer is '   0.25'"), STR("answer is '{:>7.6}'"), F(0.25));
  check(STR("answer is '0.25   '"), STR("answer is '{:<7.6}'"), F(0.25));
  check(STR("answer is ' 0.25  '"), STR("answer is '{:^7.6}'"), F(0.25));

  check(STR("answer is '---0.125'"), STR("answer is '{:->8.6}'"), F(125e-3));
  check(STR("answer is '0.125---'"), STR("answer is '{:-<8.6}'"), F(125e-3));
  check(STR("answer is '-0.125--'"), STR("answer is '{:-^8.6}'"), F(125e-3));

  check(STR("answer is '***inf'"), STR("answer is '{:*>6.6}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf***'"), STR("answer is '{:*<6.6}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '*inf**'"), STR("answer is '{:*^6.6}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '###-inf'"), STR("answer is '{:#>7.6}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf###'"), STR("answer is '{:#<7.6}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '#-inf##'"), STR("answer is '{:#^7.6}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '^^^nan'"), STR("answer is '{:^>6.6}'"), nan_pos);
  check(STR("answer is 'nan^^^'"), STR("answer is '{:^<6.6}'"), nan_pos);
  check(STR("answer is '^nan^^'"), STR("answer is '{:^^6.6}'"), nan_pos);

  check(STR("answer is '000-nan'"), STR("answer is '{:0>7.6}'"), nan_neg);
  check(STR("answer is '-nan000'"), STR("answer is '{:0<7.6}'"), nan_neg);
  check(STR("answer is '0-nan00'"), STR("answer is '{:0^7.6}'"), nan_neg);

  // Test whether zero padding is ignored
  check(STR("answer is '   0.25'"), STR("answer is '{:>07.6}'"), F(0.25));
  check(STR("answer is '0.25   '"), STR("answer is '{:<07.6}'"), F(0.25));
  check(STR("answer is ' 0.25  '"), STR("answer is '{:^07.6}'"), F(0.25));

  // *** Sign ***
  check(STR("answer is '0'"), STR("answer is '{:.6}'"), F(0));
  check(STR("answer is '0'"), STR("answer is '{:-.6}'"), F(0));
  check(STR("answer is '+0'"), STR("answer is '{:+.6}'"), F(0));
  check(STR("answer is ' 0'"), STR("answer is '{: .6}'"), F(0));

  check(STR("answer is '-0'"), STR("answer is '{:.6}'"), F(-0.));
  check(STR("answer is '-0'"), STR("answer is '{:-.6}'"), F(-0.));
  check(STR("answer is '-0'"), STR("answer is '{:+.6}'"), F(-0.));
  check(STR("answer is '-0'"), STR("answer is '{: .6}'"), F(-0.));

  // [format.string.std]/5 The sign option applies to floating-point infinity and NaN.
  check(STR("answer is 'inf'"), STR("answer is '{:.6}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is 'inf'"), STR("answer is '{:-.6}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '+inf'"), STR("answer is '{:+.6}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is ' inf'"), STR("answer is '{: .6}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '-inf'"), STR("answer is '{:.6}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:-.6}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:+.6}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{: .6}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:.6}'"), nan_pos);
  check(STR("answer is 'nan'"), STR("answer is '{:-.6}'"), nan_pos);
  check(STR("answer is '+nan'"), STR("answer is '{:+.6}'"), nan_pos);
  check(STR("answer is ' nan'"), STR("answer is '{: .6}'"), nan_pos);

  check(STR("answer is '-nan'"), STR("answer is '{:.6}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:-.6}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{:+.6}'"), nan_neg);
  check(STR("answer is '-nan'"), STR("answer is '{: .6}'"), nan_neg);

  // *** alternate form **
  // When precision is zero there's no decimal point except when the alternate form is specified.
  check(STR("answer is '0'"), STR("answer is '{:.0}'"), F(0));
  check(STR("answer is '0.'"), STR("answer is '{:#.0}'"), F(0));

  check(STR("answer is '0.'"), STR("answer is '{:#.6}'"), F(0));
  check(STR("answer is '2.5'"), STR("answer is '{:#.6}'"), F(2.5));

  check(STR("answer is 'inf'"), STR("answer is '{:#.6}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '-inf'"), STR("answer is '{:#.6}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is 'nan'"), STR("answer is '{:#.6}'"), nan_pos);
  check(STR("answer is '-nan'"), STR("answer is '{:#.6}'"), nan_neg);

  // *** zero-padding & width ***
  check(STR("answer is '0.03125'"), STR("answer is '{:06.6}'"), 0.03125);
  check(STR("answer is '+0.03125'"), STR("answer is '{:+06.6}'"), 0.03125);
  check(STR("answer is '+0.03125'"), STR("answer is '{:+07.6}'"), 0.03125);
  check(STR("answer is '+0.03125'"), STR("answer is '{:+08.6}'"), 0.03125);

  check(STR("answer is '000.03125'"), STR("answer is '{:09.6}'"), 0.03125);
  check(STR("answer is '000.03125'"), STR("answer is '{:-09.6}'"), 0.03125);
  check(STR("answer is '+00.03125'"), STR("answer is '{:+09.6}'"), 0.03125);
  check(STR("answer is ' 00.03125'"), STR("answer is '{: 09.6}'"), 0.03125);

  check(STR("answer is '       inf'"), STR("answer is '{:010.6}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{:-010.6}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '      +inf'"), STR("answer is '{:+010.6}'"), std::numeric_limits<F>::infinity());
  check(STR("answer is '       inf'"), STR("answer is '{: 010.6}'"), std::numeric_limits<F>::infinity());

  check(STR("answer is '      -inf'"), STR("answer is '{:010.6}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:-010.6}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{:+010.6}'"), -std::numeric_limits<F>::infinity());
  check(STR("answer is '      -inf'"), STR("answer is '{: 010.6}'"), -std::numeric_limits<F>::infinity());

  check(STR("answer is '       nan'"), STR("answer is '{:010.6}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{:-010.6}'"), nan_pos);
  check(STR("answer is '      +nan'"), STR("answer is '{:+010.6}'"), nan_pos);
  check(STR("answer is '       nan'"), STR("answer is '{: 010.6}'"), nan_pos);

  check(STR("answer is '      -nan'"), STR("answer is '{:010.6}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:-010.6}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{:+010.6}'"), nan_neg);
  check(STR("answer is '      -nan'"), STR("answer is '{: 010.6}'"), nan_neg);

  // *** precision ***
  check(STR("answer is '0.03'"), STR("answer is '{:.0}'"), 0.03125);
  check(STR("answer is '0.03'"), STR("answer is '{:.1}'"), 0.03125);
  check(STR("answer is '0.031'"), STR("answer is '{:.2}'"), 0.03125);
  check(STR("answer is '0.0312'"), STR("answer is '{:.3}'"), 0.03125);
  check(STR("answer is '0.03125'"), STR("answer is '{:.4}'"), 0.03125);
  check(STR("answer is '0.03125'"), STR("answer is '{:.5}'"), 0.03125);
  check(STR("answer is '0.03125'"), STR("answer is '{:.10}'"), 0.03125);

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
  check(STR("answer is '   0x0'"), STR("answer is '{:6}'"), P(nullptr));
  check(STR("answer is '   0x0'"), STR("answer is '{:>6}'"), P(nullptr));
  check(STR("answer is '0x0   '"), STR("answer is '{:<6}'"), P(nullptr));
  check(STR("answer is ' 0x0  '"), STR("answer is '{:^6}'"), P(nullptr));

  check(STR("answer is '---0x0'"), STR("answer is '{:->6}'"), P(nullptr));
  check(STR("answer is '0x0---'"), STR("answer is '{:-<6}'"), P(nullptr));
  check(STR("answer is '-0x0--'"), STR("answer is '{:-^6}'"), P(nullptr));

  // *** Sign ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:-}"), P(nullptr));
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:+}"), P(nullptr));
  check_exception("The format-spec should consume the input or end with a '}'", STR("{: }"), P(nullptr));

  // *** alternate form ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:#}"), P(nullptr));

  // *** zero-padding ***
  check_exception("A format-spec width field shouldn't have a leading zero", STR("{:0}"), P(nullptr));

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:.}"), P(nullptr));

  // *** locale-specific form ***
  check_exception("The format-spec should consume the input or end with a '}'", STR("{:L}"), P(nullptr));

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("p"))
    check_exception("The format-spec type has a type not supported for a pointer argument", fmt, P(nullptr));
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_handle(TestFunction check, ExceptionTest check_exception) {
  // *** Valid permuatations ***
  check(STR("answer is '0xaaaa'"), STR("answer is '{}'"), status::foo);
  check(STR("answer is '0xaaaa'"), STR("answer is '{:x}'"), status::foo);
  check(STR("answer is '0XAAAA'"), STR("answer is '{:X}'"), status::foo);
  check(STR("answer is 'foo'"), STR("answer is '{:s}'"), status::foo);

  check(STR("answer is '0x5555'"), STR("answer is '{}'"), status::bar);
  check(STR("answer is '0x5555'"), STR("answer is '{:x}'"), status::bar);
  check(STR("answer is '0X5555'"), STR("answer is '{:X}'"), status::bar);
  check(STR("answer is 'bar'"), STR("answer is '{:s}'"), status::bar);

  check(STR("answer is '0xaa55'"), STR("answer is '{}'"), status::foobar);
  check(STR("answer is '0xaa55'"), STR("answer is '{:x}'"), status::foobar);
  check(STR("answer is '0XAA55'"), STR("answer is '{:X}'"), status::foobar);
  check(STR("answer is 'foobar'"), STR("answer is '{:s}'"), status::foobar);

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

template <class CharT, class TestFunction, class ExceptionTest>
void format_tests(TestFunction check, ExceptionTest check_exception) {
  // *** Test escaping  ***
  check(STR("{"), STR("{{"));
  check(STR("}"), STR("}}"));

  // *** Test argument ID ***
  check(STR("hello false true"), STR("hello {0:} {1:}"), false, true);
  check(STR("hello true false"), STR("hello {1:} {0:}"), false, true);

  // ** Test invalid format strings ***
  check_exception("The format string terminates at a '{'", STR("{"));
  check_exception("The replacement field misses a terminating '}'", STR("{:"), 42);

  check_exception("The format string contains an invalid escape sequence", STR("}"));
  check_exception("The format string contains an invalid escape sequence", STR("{:}-}"), 42);

  check_exception("The format string contains an invalid escape sequence", STR("} "));

  check_exception("The arg-id of the format-spec starts with an invalid character", STR("{-"), 42);
  check_exception("Argument index out of bounds", STR("hello {}"));
  check_exception("Argument index out of bounds", STR("hello {0}"));
  check_exception("Argument index out of bounds", STR("hello {1}"), 42);

  // *** Test char format argument ***
  // The `char` to `wchar_t` formatting is tested separately.
  check(STR("hello 09azAZ!"), STR("hello {}{}{}{}{}{}{}"), CharT('0'), CharT('9'), CharT('a'), CharT('z'), CharT('A'),
        CharT('Z'), CharT('!'));

  format_test_char<CharT>(check, check_exception);
  format_test_char_as_integer<CharT>(check, check_exception);

  // *** Test string format argument ***
  {
    CharT buffer[] = {CharT('0'), CharT('9'), CharT('a'), CharT('z'), CharT('A'), CharT('Z'), CharT('!'), 0};
    CharT* data = buffer;
    check(STR("hello 09azAZ!"), STR("hello {}"), data);
  }
  {
    CharT buffer[] = {CharT('0'), CharT('9'), CharT('a'), CharT('z'), CharT('A'), CharT('Z'), CharT('!'), 0};
    const CharT* data = buffer;
    check(STR("hello 09azAZ!"), STR("hello {}"), data);
  }
  {
    std::basic_string<CharT> data = STR("world");
    check(STR("hello world"), STR("hello {}"), data);
  }
  {
    std::basic_string<CharT> buffer = STR("world");
    std::basic_string_view<CharT> data = buffer;
    check(STR("hello world"), STR("hello {}"), data);
  }
  format_string_tests<CharT>(check, check_exception);

  // *** Test Boolean format argument ***
  check(STR("hello false true"), STR("hello {} {}"), false, true);

  format_test_bool<CharT>(check, check_exception);
  format_test_bool_as_char<CharT>(check, check_exception);
  format_test_bool_as_integer<CharT>(check, check_exception);

  // *** Test signed integral format argument ***
  check(STR("hello 42"), STR("hello {}"), static_cast<signed char>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<short>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<int>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<long>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<long long>(42));
#ifndef TEST_HAS_NO_INT128
  check(STR("hello 42"), STR("hello {}"), static_cast<__int128_t>(42));
  {
    // Note 128-bit support is only partly implemented test the range
    // conditions here.
    std::basic_string<CharT> min = std::format(STR("{}"), std::numeric_limits<long long>::min());
    check(min, STR("{}"), static_cast<__int128_t>(std::numeric_limits<long long>::min()));
    std::basic_string<CharT> max = std::format(STR("{}"), std::numeric_limits<long long>::max());
    check(max, STR("{}"), static_cast<__int128_t>(std::numeric_limits<long long>::max()));
    check_exception("128-bit value is outside of implemented range", STR("{}"),
                    static_cast<__int128_t>(std::numeric_limits<long long>::min()) - 1);
    check_exception("128-bit value is outside of implemented range", STR("{}"),
                    static_cast<__int128_t>(std::numeric_limits<long long>::max()) + 1);
  }
#endif
  format_test_signed_integer<CharT>(check, check_exception);

  // ** Test unsigned integral format argument ***
  check(STR("hello 42"), STR("hello {}"), static_cast<unsigned char>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<unsigned short>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<unsigned>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<unsigned long>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<unsigned long long>(42));
#ifndef TEST_HAS_NO_INT128
  check(STR("hello 42"), STR("hello {}"), static_cast<__uint128_t>(42));
  {
    // Note 128-bit support is only partly implemented test the range
    // conditions here.
    std::basic_string<CharT> max = std::format(STR("{}"), std::numeric_limits<unsigned long long>::max());
    check(max, STR("{}"), static_cast<__uint128_t>(std::numeric_limits<unsigned long long>::max()));
    check_exception("128-bit value is outside of implemented range", STR("{}"),
                    static_cast<__uint128_t>(std::numeric_limits<unsigned long long>::max()) + 1);
  }
#endif
  format_test_unsigned_integer<CharT>(check, check_exception);

  // *** Test floating point format argument ***
  check(STR("hello 42"), STR("hello {}"), static_cast<float>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<double>(42));
  check(STR("hello 42"), STR("hello {}"), static_cast<long double>(42));
  format_test_floating_point<CharT>(check, check_exception);

  // *** Test pointer formater argument ***
  check(STR("hello 0x0"), STR("hello {}"), nullptr);
  check(STR("hello 0x42"), STR("hello {}"), reinterpret_cast<void*>(0x42));
  check(STR("hello 0x42"), STR("hello {}"), reinterpret_cast<const void*>(0x42));
  format_test_pointer<CharT>(check, check_exception);

  // *** Test handle formatter argument ***
  format_test_handle<CharT>(check, check_exception);
}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
template <class TestFunction>
void format_tests_char_to_wchar_t(TestFunction check) {
  using CharT = wchar_t;
  check(STR("hello 09azA"), STR("hello {}{}{}{}{}"), '0', '9', 'a', 'z', 'A');
}
#endif

#endif
