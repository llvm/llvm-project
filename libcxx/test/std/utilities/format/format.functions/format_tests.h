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
void format_tests(TestFunction check, ExceptionTest check_exception) {
  // *** Test escaping  ***
  check(STR("{"), STR("{{"));
  check(STR("}"), STR("}}"));

  // *** Test argument ID ***
  check(STR("hello 01"), STR("hello {0:}{1:}"), false, true);
  check(STR("hello 10"), STR("hello {1:}{0:}"), false, true);

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
  check(STR("hello 01"), STR("hello {}{}"), false, true);

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
