//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_UTILITIES_FORMAT_FORMAT_FUNCTIONS_FORMAT_TESTS_H
#define TEST_STD_UTILITIES_FORMAT_FORMAT_FUNCTIONS_FORMAT_TESTS_H

#include "make_string.h"

// In this file the following template types are used:
// TestFunction must be callable as check(expected-result, string-to-format, args-to-format...)
// ExceptionTest must be callable as check_exception(expected-exception, string-to-format, args-to-format...)

#define STR(S) MAKE_STRING(CharT, S)
#define CSTR(S) MAKE_CSTRING(CharT, S)

template <class CharT>
std::vector<std::basic_string<CharT>> invalid_types(std::string valid) {
  std::vector<std::basic_string<CharT>> result;

#define CASE(T)                                                                \
  case #T[0]:                                                                  \
    result.push_back(STR("Invalid formatter type {:" #T "}"));                 \
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
void format_test_string(T world, T universe, TestFunction check,
                        ExceptionTest check_exception) {

  // *** Valid input tests ***
  // Unsed argument is ignored. TODO FMT what does the Standard mandate?
  check(STR("hello world"), STR("hello {}"), world, universe);
  check(STR("hello world and universe"), STR("hello {} and {}"), world,
        universe);
  check(STR("hello world"), STR("hello {0}"), world, universe);
  check(STR("hello universe"), STR("hello {1}"), world, universe);
  check(STR("hello universe and world"), STR("hello {1} and {0}"), world,
        universe);

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
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("hello {:-}"), world);

  // *** alternate form ***
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("hello {:#}"), world);

  // *** zero-padding ***
  check_exception("A format-spec width field shouldn't have a leading zero",
                  STR("hello {:0}"), world);

  // *** width ***
#if _LIBCPP_VERSION
  // This limit isn't specified in the Standard.
  static_assert(std::__format::__number_max == 2'147'483'647,
                "Update the assert and the test.");
  check_exception("The numeric value of the format-spec is too large",
                  STR("{:2147483648}"), world);
  check_exception("The numeric value of the format-spec is too large",
                  STR("{:5000000000}"), world);
  check_exception("The numeric value of the format-spec is too large",
                  STR("{:10000000000}"), world);
#endif

  check_exception(
      "A format-spec width field replacement should have a positive value",
      STR("hello {:{}}"), world, 0);
  check_exception(
      "A format-spec arg-id replacement shouldn't have a negative value",
      STR("hello {:{}}"), world, -1);
  check_exception(
      "A format-spec arg-id replacement exceeds the maximum supported value",
      STR("hello {:{}}"), world, -1u);
  check_exception("Argument index out of bounds", STR("hello {:{}}"), world);
  check_exception(
      "A format-spec arg-id replacement argument isn't an integral type",
      STR("hello {:{}}"), world, universe);
  check_exception(
      "Using manual argument numbering in automatic argument numbering mode",
      STR("hello {:{0}}"), world, 1);
  check_exception(
      "Using automatic argument numbering in manual argument numbering mode",
      STR("hello {0:{}}"), world, 1);

  // *** precision ***
  check_exception("A format-spec precision field shouldn't have a leading zero",
                  STR("hello {:.01}"), world);

#if _LIBCPP_VERSION
  // This limit isn't specified in the Standard.
  static_assert(std::__format::__number_max == 2'147'483'647,
                "Update the assert and the test.");
  check_exception("The numeric value of the format-spec is too large",
                  STR("{:.2147483648}"), world);
  check_exception("The numeric value of the format-spec is too large",
                  STR("{:.5000000000}"), world);
  check_exception("The numeric value of the format-spec is too large",
                  STR("{:.10000000000}"), world);
#endif

  // Precision 0 allowed, but not useful for string arguments.
  check(STR("hello "), STR("hello {:.{}}"), world, 0);
  check_exception(
      "A format-spec arg-id replacement shouldn't have a negative value",
      STR("hello {:.{}}"), world, -1);
  check_exception(
      "A format-spec arg-id replacement exceeds the maximum supported value",
      STR("hello {:.{}}"), world, -1u);
  check_exception("Argument index out of bounds", STR("hello {:.{}}"), world);
  check_exception(
      "A format-spec arg-id replacement argument isn't an integral type",
      STR("hello {:.{}}"), world, universe);
  check_exception(
      "Using manual argument numbering in automatic argument numbering mode",
      STR("hello {:.{0}}"), world, 1);
  check_exception(
      "Using automatic argument numbering in manual argument numbering mode",
      STR("hello {0:.{}}"), world, 1);

  // *** locale-specific form ***
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("hello {:L}"), world);

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("s"))
    check_exception(
        "The format-spec type has a type not supported for a string argument",
        fmt, world);
}

template <class CharT, class TestFunction>
void format_test_string_unicode(TestFunction check) {
#ifndef _LIBCPP_HAS_NO_UNICODE
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
#else
  (void)check;
#endif
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_string_tests(TestFunction check, ExceptionTest check_exception) {
  std::basic_string<CharT> world = STR("world");
  std::basic_string<CharT> universe = STR("universe");

  // Testing the char const[] is a bit tricky due to array to pointer decay.
  // Since there are separate tests in format.formatter.spec the array is not
  // tested here.
  format_test_string<CharT>(world.c_str(), universe.c_str(), check,
                            check_exception);
  format_test_string<CharT>(const_cast<CharT*>(world.c_str()),
                            const_cast<CharT*>(universe.c_str()), check,
                            check_exception);
  format_test_string<CharT>(std::basic_string_view<CharT>(world),
                            std::basic_string_view<CharT>(universe), check,
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
  check_exception("A sign field isn't allowed in this format-spec", STR("{:-}"),
                  true);
  check_exception("A sign field isn't allowed in this format-spec", STR("{:+}"),
                  true);
  check_exception("A sign field isn't allowed in this format-spec", STR("{: }"),
                  true);

  check_exception("A sign field isn't allowed in this format-spec",
                  STR("{:-s}"), true);
  check_exception("A sign field isn't allowed in this format-spec",
                  STR("{:+s}"), true);
  check_exception("A sign field isn't allowed in this format-spec",
                  STR("{: s}"), true);

  // *** alternate form ***
  check_exception("An alternate form field isn't allowed in this format-spec",
                  STR("{:#}"), true);
  check_exception("An alternate form field isn't allowed in this format-spec",
                  STR("{:#s}"), true);

  // *** zero-padding ***
  check_exception("A zero-padding field isn't allowed in this format-spec",
                  STR("{:0}"), true);
  check_exception("A zero-padding field isn't allowed in this format-spec",
                  STR("{:0s}"), true);

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.}"), true);
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.0}"), true);
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.42}"), true);

  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.s}"), true);
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.0s}"), true);
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.42s}"), true);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdosxX"))
    check_exception(
        "The format-spec type has a type not supported for a bool argument",
        fmt, true);
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_bool_as_char(TestFunction check,
                              ExceptionTest check_exception) {
  // *** align-fill & width ***
  check(STR("answer is '\1     '"), STR("answer is '{:6c}'"), true);
  check(STR("answer is '     \1'"), STR("answer is '{:>6c}'"), true);
  check(STR("answer is '\1     '"), STR("answer is '{:<6c}'"), true);
  check(STR("answer is '  \1   '"), STR("answer is '{:^6c}'"), true);

  check(STR("answer is '-----\1'"), STR("answer is '{:->6c}'"), true);
  check(STR("answer is '\1-----'"), STR("answer is '{:-<6c}'"), true);
  check(STR("answer is '--\1---'"), STR("answer is '{:-^6c}'"), true);

  check(std::basic_string<CharT>(CSTR("answer is '\0     '"), 18),
        STR("answer is '{:6c}'"), false);
  check(std::basic_string<CharT>(CSTR("answer is '\0     '"), 18),
        STR("answer is '{:6c}'"), false);
  check(std::basic_string<CharT>(CSTR("answer is '     \0'"), 18),
        STR("answer is '{:>6c}'"), false);
  check(std::basic_string<CharT>(CSTR("answer is '\0     '"), 18),
        STR("answer is '{:<6c}'"), false);
  check(std::basic_string<CharT>(CSTR("answer is '  \0   '"), 18),
        STR("answer is '{:^6c}'"), false);

  check(std::basic_string<CharT>(CSTR("answer is '-----\0'"), 18),
        STR("answer is '{:->6c}'"), false);
  check(std::basic_string<CharT>(CSTR("answer is '\0-----'"), 18),
        STR("answer is '{:-<6c}'"), false);
  check(std::basic_string<CharT>(CSTR("answer is '--\0---'"), 18),
        STR("answer is '{:-^6c}'"), false);

  // *** Sign ***
  check_exception("A sign field isn't allowed in this format-spec",
                  STR("{:-c}"), true);
  check_exception("A sign field isn't allowed in this format-spec",
                  STR("{:+c}"), true);
  check_exception("A sign field isn't allowed in this format-spec",
                  STR("{: c}"), true);

  // *** alternate form ***
  check_exception("An alternate form field isn't allowed in this format-spec",
                  STR("{:#c}"), true);

  // *** zero-padding ***
  check_exception("A zero-padding field isn't allowed in this format-spec",
                  STR("{:0c}"), true);

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.c}"), true);
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.0c}"), true);
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.42c}"), true);

  // *** locale-specific form ***
  // Note it has no effect but it's allowed.
  check(STR("answer is '*'"), STR("answer is '{:Lc}'"), '*');

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdosxX"))
    check_exception(
        "The format-spec type has a type not supported for a bool argument",
        fmt, true);
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_bool_as_integer(TestFunction check,
                                 ExceptionTest check_exception) {
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
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.}"), true);
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.0}"), true);
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.42}"), true);

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdosxX"))
    check_exception(
        "The format-spec type has a type not supported for a bool argument",
        fmt, true);
}

template <class I, class CharT, class TestFunction, class ExceptionTest>
void format_test_integer_as_integer(TestFunction check,
                                    ExceptionTest check_exception) {
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
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.}"), I(0));
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.0}"), I(0));
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.42}"), I(0));

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
    check_exception(
        "The format-spec type has a type not supported for an integer argument",
        fmt, 42);
}

template <class I, class CharT, class TestFunction, class ExceptionTest>
void format_test_integer_as_char(TestFunction check,
                                 ExceptionTest check_exception) {
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
  check_exception("A sign field isn't allowed in this format-spec",
                  STR("answer is {:-c}"), I(42));
  check_exception("A sign field isn't allowed in this format-spec",
                  STR("answer is {:+c}"), I(42));
  check_exception("A sign field isn't allowed in this format-spec",
                  STR("answer is {: c}"), I(42));

  // *** alternate form ***
  check_exception("An alternate form field isn't allowed in this format-spec",
                  STR("answer is {:#c}"), I(42));

  // *** zero-padding & width ***
  check_exception("A zero-padding field isn't allowed in this format-spec",
                  STR("answer is {:01c}"), I(42));

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.c}"), I(0));
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.0c}"), I(0));
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.42c}"), I(0));

  // *** locale-specific form ***
  // Note it has no effect but it's allowed.
  check(STR("answer is '*'"), STR("answer is '{:Lc}'"), I(42));

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
    check_exception(
        "The format-spec type has a type not supported for an integer argument",
        fmt, I(42));

  // *** Validate range ***
  // TODO FMT Update test after adding 128-bit support.
  if constexpr (sizeof(I) <= sizeof(long long)) {
    // The code has some duplications to keep the if statement readable.
    if constexpr (std::signed_integral<CharT>) {
      if constexpr (std::signed_integral<I> && sizeof(I) > sizeof(CharT)) {
        check_exception("Integral value outside the range of the char type",
                        STR("{:c}"), std::numeric_limits<I>::min());
        check_exception("Integral value outside the range of the char type",
                        STR("{:c}"), std::numeric_limits<I>::max());
      } else if constexpr (std::unsigned_integral<I> &&
                           sizeof(I) >= sizeof(CharT)) {
        check_exception("Integral value outside the range of the char type",
                        STR("{:c}"), std::numeric_limits<I>::max());
      }
    } else if constexpr (sizeof(I) > sizeof(CharT)) {
      check_exception("Integral value outside the range of the char type",
                      STR("{:c}"), std::numeric_limits<I>::max());
    }
  }
}

template <class I, class CharT, class TestFunction, class ExceptionTest>
void format_test_integer(TestFunction check, ExceptionTest check_exception) {
  format_test_integer_as_integer<I, CharT>(check, check_exception);
  format_test_integer_as_char<I, CharT>(check, check_exception);
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_signed_integer(TestFunction check,
                                ExceptionTest check_exception) {
  format_test_integer<signed char, CharT>(check, check_exception);
  format_test_integer<short, CharT>(check, check_exception);
  format_test_integer<int, CharT>(check, check_exception);
  format_test_integer<long, CharT>(check, check_exception);
  format_test_integer<long long, CharT>(check, check_exception);
#ifndef _LIBCPP_HAS_NO_INT128
  format_test_integer<__int128_t, CharT>(check, check_exception);
#endif
  // *** check the minma and maxima ***
  check(STR("-0b10000000"), STR("{:#b}"), std::numeric_limits<int8_t>::min());
  check(STR("-0200"), STR("{:#o}"), std::numeric_limits<int8_t>::min());
  check(STR("-128"), STR("{:#}"), std::numeric_limits<int8_t>::min());
  check(STR("-0x80"), STR("{:#x}"), std::numeric_limits<int8_t>::min());

  check(STR("-0b1000000000000000"), STR("{:#b}"),
        std::numeric_limits<int16_t>::min());
  check(STR("-0100000"), STR("{:#o}"), std::numeric_limits<int16_t>::min());
  check(STR("-32768"), STR("{:#}"), std::numeric_limits<int16_t>::min());
  check(STR("-0x8000"), STR("{:#x}"), std::numeric_limits<int16_t>::min());

  check(STR("-0b10000000000000000000000000000000"), STR("{:#b}"),
        std::numeric_limits<int32_t>::min());
  check(STR("-020000000000"), STR("{:#o}"),
        std::numeric_limits<int32_t>::min());
  check(STR("-2147483648"), STR("{:#}"), std::numeric_limits<int32_t>::min());
  check(STR("-0x80000000"), STR("{:#x}"), std::numeric_limits<int32_t>::min());

  check(STR("-0b100000000000000000000000000000000000000000000000000000000000000"
            "0"),
        STR("{:#b}"), std::numeric_limits<int64_t>::min());
  check(STR("-01000000000000000000000"), STR("{:#o}"),
        std::numeric_limits<int64_t>::min());
  check(STR("-9223372036854775808"), STR("{:#}"),
        std::numeric_limits<int64_t>::min());
  check(STR("-0x8000000000000000"), STR("{:#x}"),
        std::numeric_limits<int64_t>::min());

  check(STR("0b1111111"), STR("{:#b}"), std::numeric_limits<int8_t>::max());
  check(STR("0177"), STR("{:#o}"), std::numeric_limits<int8_t>::max());
  check(STR("127"), STR("{:#}"), std::numeric_limits<int8_t>::max());
  check(STR("0x7f"), STR("{:#x}"), std::numeric_limits<int8_t>::max());

  check(STR("0b111111111111111"), STR("{:#b}"),
        std::numeric_limits<int16_t>::max());
  check(STR("077777"), STR("{:#o}"), std::numeric_limits<int16_t>::max());
  check(STR("32767"), STR("{:#}"), std::numeric_limits<int16_t>::max());
  check(STR("0x7fff"), STR("{:#x}"), std::numeric_limits<int16_t>::max());

  check(STR("0b1111111111111111111111111111111"), STR("{:#b}"),
        std::numeric_limits<int32_t>::max());
  check(STR("017777777777"), STR("{:#o}"), std::numeric_limits<int32_t>::max());
  check(STR("2147483647"), STR("{:#}"), std::numeric_limits<int32_t>::max());
  check(STR("0x7fffffff"), STR("{:#x}"), std::numeric_limits<int32_t>::max());

  check(
      STR("0b111111111111111111111111111111111111111111111111111111111111111"),
      STR("{:#b}"), std::numeric_limits<int64_t>::max());
  check(STR("0777777777777777777777"), STR("{:#o}"),
        std::numeric_limits<int64_t>::max());
  check(STR("9223372036854775807"), STR("{:#}"),
        std::numeric_limits<int64_t>::max());
  check(STR("0x7fffffffffffffff"), STR("{:#x}"),
        std::numeric_limits<int64_t>::max());

  // TODO FMT Add __int128_t test after implementing full range.
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_unsigned_integer(TestFunction check,
                                  ExceptionTest check_exception) {
  format_test_integer<unsigned char, CharT>(check, check_exception);
  format_test_integer<unsigned short, CharT>(check, check_exception);
  format_test_integer<unsigned, CharT>(check, check_exception);
  format_test_integer<unsigned long, CharT>(check, check_exception);
  format_test_integer<unsigned long long, CharT>(check, check_exception);
#ifndef _LIBCPP_HAS_NO_INT128
  format_test_integer<__uint128_t, CharT>(check, check_exception);
#endif
  // *** test the maxima ***
  check(STR("0b11111111"), STR("{:#b}"), std::numeric_limits<uint8_t>::max());
  check(STR("0377"), STR("{:#o}"), std::numeric_limits<uint8_t>::max());
  check(STR("255"), STR("{:#}"), std::numeric_limits<uint8_t>::max());
  check(STR("0xff"), STR("{:#x}"), std::numeric_limits<uint8_t>::max());

  check(STR("0b1111111111111111"), STR("{:#b}"),
        std::numeric_limits<uint16_t>::max());
  check(STR("0177777"), STR("{:#o}"), std::numeric_limits<uint16_t>::max());
  check(STR("65535"), STR("{:#}"), std::numeric_limits<uint16_t>::max());
  check(STR("0xffff"), STR("{:#x}"), std::numeric_limits<uint16_t>::max());

  check(STR("0b11111111111111111111111111111111"), STR("{:#b}"),
        std::numeric_limits<uint32_t>::max());
  check(STR("037777777777"), STR("{:#o}"),
        std::numeric_limits<uint32_t>::max());
  check(STR("4294967295"), STR("{:#}"), std::numeric_limits<uint32_t>::max());
  check(STR("0xffffffff"), STR("{:#x}"), std::numeric_limits<uint32_t>::max());

  check(
      STR("0b1111111111111111111111111111111111111111111111111111111111111111"),
      STR("{:#b}"), std::numeric_limits<uint64_t>::max());
  check(STR("01777777777777777777777"), STR("{:#o}"),
        std::numeric_limits<uint64_t>::max());
  check(STR("18446744073709551615"), STR("{:#}"),
        std::numeric_limits<uint64_t>::max());
  check(STR("0xffffffffffffffff"), STR("{:#x}"),
        std::numeric_limits<uint64_t>::max());

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
  check_exception("A sign field isn't allowed in this format-spec", STR("{:-}"),
                  CharT('*'));
  check_exception("A sign field isn't allowed in this format-spec", STR("{:+}"),
                  CharT('*'));
  check_exception("A sign field isn't allowed in this format-spec", STR("{: }"),
                  CharT('*'));

  check_exception("A sign field isn't allowed in this format-spec",
                  STR("{:-c}"), CharT('*'));
  check_exception("A sign field isn't allowed in this format-spec",
                  STR("{:+c}"), CharT('*'));
  check_exception("A sign field isn't allowed in this format-spec",
                  STR("{: c}"), CharT('*'));

  // *** alternate form ***
  check_exception("An alternate form field isn't allowed in this format-spec",
                  STR("{:#}"), CharT('*'));
  check_exception("An alternate form field isn't allowed in this format-spec",
                  STR("{:#c}"), CharT('*'));

  // *** zero-padding ***
  check_exception("A zero-padding field isn't allowed in this format-spec",
                  STR("{:0}"), CharT('*'));
  check_exception("A zero-padding field isn't allowed in this format-spec",
                  STR("{:0c}"), CharT('*'));

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.0}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.42}"), CharT('*'));

  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.c}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.0c}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.42c}"), CharT('*'));

  // *** locale-specific form ***
  // Note it has no effect but it's allowed.
  check(STR("answer is '*'"), STR("answer is '{:L}'"), '*');
  check(STR("answer is '*'"), STR("answer is '{:Lc}'"), '*');

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
    check_exception(
        "The format-spec type has a type not supported for a char argument",
        fmt, CharT('*'));
}

template <class CharT, class TestFunction, class ExceptionTest>
void format_test_char_as_integer(TestFunction check,
                                 ExceptionTest check_exception) {
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
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.d}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.0d}"), CharT('*'));
  check_exception("The format-spec should consume the input or end with a '}'",
                  STR("{:.42d}"), CharT('*'));

  // *** locale-specific form ***
  // See locale-specific_form.pass.cpp

  // *** type ***
  for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
    check_exception(
        "The format-spec type has a type not supported for a char argument",
        fmt, '*');
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
  check_exception("The replacement field misses a terminating '}'", STR("{:"),
                  42);

  check_exception("The format string contains an invalid escape sequence",
                  STR("}"));
  check_exception("The format string contains an invalid escape sequence",
                  STR("{:}-}"), 42);

  check_exception("The format string contains an invalid escape sequence",
                  STR("} "));

  check_exception(
      "The arg-id of the format-spec starts with an invalid character",
      STR("{-"), 42);
  check_exception("Argument index out of bounds", STR("hello {}"));
  check_exception("Argument index out of bounds", STR("hello {0}"));
  check_exception("Argument index out of bounds", STR("hello {1}"), 42);

  // *** Test char format argument ***
  // The `char` to `wchar_t` formatting is tested separately.
  check(STR("hello 09azAZ!"), STR("hello {}{}{}{}{}{}{}"), CharT('0'),
        CharT('9'), CharT('a'), CharT('z'), CharT('A'), CharT('Z'), CharT('!'));

  format_test_char<CharT>(check, check_exception);
  format_test_char_as_integer<CharT>(check, check_exception);

  // *** Test string format argument ***
  {
    CharT buffer[] = {CharT('0'), CharT('9'), CharT('a'), CharT('z'),
                      CharT('A'), CharT('Z'), CharT('!'), 0};
    CharT* data = buffer;
    check(STR("hello 09azAZ!"), STR("hello {}"), data);
  }
  {
    CharT buffer[] = {CharT('0'), CharT('9'), CharT('a'), CharT('z'),
                      CharT('A'), CharT('Z'), CharT('!'), 0};
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
#ifndef _LIBCPP_HAS_NO_INT128
  check(STR("hello 42"), STR("hello {}"), static_cast<__int128_t>(42));
  {
    // Note 128-bit support is only partly implemented test the range
    // conditions here.
    std::basic_string<CharT> min =
        std::format(STR("{}"), std::numeric_limits<long long>::min());
    check(min, STR("{}"),
          static_cast<__int128_t>(std::numeric_limits<long long>::min()));
    std::basic_string<CharT> max =
        std::format(STR("{}"), std::numeric_limits<long long>::max());
    check(max, STR("{}"),
          static_cast<__int128_t>(std::numeric_limits<long long>::max()));
    check_exception(
        "128-bit value is outside of implemented range", STR("{}"),
        static_cast<__int128_t>(std::numeric_limits<long long>::min()) - 1);
    check_exception(
        "128-bit value is outside of implemented range", STR("{}"),
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
#ifndef _LIBCPP_HAS_NO_INT128
  check(STR("hello 42"), STR("hello {}"), static_cast<__uint128_t>(42));
  {
    // Note 128-bit support is only partly implemented test the range
    // conditions here.
    std::basic_string<CharT> max =
        std::format(STR("{}"), std::numeric_limits<unsigned long long>::max());
    check(max, STR("{}"),
          static_cast<__uint128_t>(
              std::numeric_limits<unsigned long long>::max()));
    check_exception("128-bit value is outside of implemented range", STR("{}"),
                    static_cast<__uint128_t>(
                        std::numeric_limits<unsigned long long>::max()) +
                        1);
  }
#endif
  format_test_unsigned_integer<CharT>(check, check_exception);

  // *** Test floating point format argument ***
// TODO FMT Enable after floating-point support has been enabled
#if 0
  check(STR("hello 42.000000"), STR("hello {}"), static_cast<float>(42));
  check(STR("hello 42.000000"), STR("hello {}"), static_cast<double>(42));
  check(STR("hello 42.000000"), STR("hello {}"), static_cast<long double>(42));
#endif
}

template <class TestFunction>
void format_tests_char_to_wchar_t(TestFunction check) {
  using CharT = wchar_t;
  check(STR("hello 09azA"), STR("hello {}{}{}{}{}"), '0', '9', 'a', 'z', 'A');
}

#endif
