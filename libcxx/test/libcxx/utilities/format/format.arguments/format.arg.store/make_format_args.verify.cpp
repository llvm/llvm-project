//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// template<class Context = format_context, class... Args>
// format-arg-store<Context, Args...> make_format_args(Args&... args);
//
// Preconditions: The type typename Context::template formatter_type<remove_const_t<Ti>>
// meets the BasicFormatter requirements ([formatter.requirements]) for each Ti in Args.
//
// When the precondition is violated libc++ diagnoses the issue with the
// formatter specialization.

#include <format>

#include "test_macros.h"

struct no_formatter_specialization {};
void test_no_formatter_specialization() {
  no_formatter_specialization t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization has not been provided.}}
  (void)std::make_format_args(t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization has not been provided.}}
  (void)std::make_wformat_args(t);
#endif
}

struct correct_formatter_specialization {};
template <class CharT>
struct std::formatter<correct_formatter_specialization, CharT> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext&);

  template <class FormatContext>
  typename FormatContext::iterator format(correct_formatter_specialization&, FormatContext&) const;
};
void test_correct_formatter_specialization() {
  correct_formatter_specialization t;
  (void)std::make_format_args(t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  (void)std::make_wformat_args(t);
#endif
}

struct formatter_not_semiregular {};
template <class CharT>
struct std::formatter<formatter_not_semiregular, CharT> {
  formatter(int);

  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext&);

  template <class FormatContext>
  typename FormatContext::iterator format(formatter_not_semiregular&, FormatContext&) const;
};
void test_formatter_not_semiregular() {
  formatter_not_semiregular t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization is not semiregular.}}
  (void)std::make_format_args(t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization is not semiregular.}}
  (void)std::make_wformat_args(t);
#endif
}

struct formatter_no_parse_function {};
template <class CharT>
struct std::formatter<formatter_no_parse_function, CharT> {
  template <class FormatContext>
  typename FormatContext::iterator format(formatter_no_parse_function&, FormatContext&) const;
};
void test_formatter_no_parse_function() {
  formatter_no_parse_function t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a parse function taking the proper arguments.}}
  (void)std::make_format_args(t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a parse function taking the proper arguments.}}
  (void)std::make_wformat_args(t);
#endif
}

struct parse_function_invalid_arguments {};
template <class CharT>
struct std::formatter<parse_function_invalid_arguments, CharT> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext&, int);

  template <class FormatContext>
  typename FormatContext::iterator format(parse_function_invalid_arguments&, FormatContext&) const;
};
void test_parse_function_invalid_arguments() {
  parse_function_invalid_arguments t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a parse function taking the proper arguments.}}
  (void)std::make_format_args(t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a parse function taking the proper arguments.}}
  (void)std::make_wformat_args(t);
#endif
}

struct parse_function_invalid_return_type {};
template <class CharT>
struct std::formatter<parse_function_invalid_return_type, CharT> {
  template <class ParseContext>
  constexpr int parse(ParseContext&);

  template <class FormatContext>
  typename FormatContext::iterator format(parse_function_invalid_return_type&, FormatContext&) const;
};
void test_parse_function_invalid_return_type() {
  parse_function_invalid_return_type t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization's parse function does not return the required type.}}
  (void)std::make_format_args(t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization's parse function does not return the required type.}}
  (void)std::make_wformat_args(t);
#endif
}

struct no_format_function {};
template <class CharT>
struct std::formatter<no_format_function, CharT> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext&);
};
void test_no_format_function() {
  no_format_function t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a format function taking the proper arguments.}}
  (void)std::make_format_args(t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a format function taking the proper arguments.}}
  (void)std::make_wformat_args(t);
#endif
}

struct format_function_invalid_arguments {};
template <class CharT>
struct std::formatter<format_function_invalid_arguments, CharT> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext&);

  template <class FormatContext>
  typename FormatContext::iterator format(format_function_invalid_arguments&) const;
};
void test_format_function_invalid_arguments() {
  format_function_invalid_arguments t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a format function taking the proper arguments.}}
  (void)std::make_format_args(t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a format function taking the proper arguments.}}
  (void)std::make_wformat_args(t);
#endif
}

struct format_function_invalid_return_type {};
template <class CharT>
struct std::formatter<format_function_invalid_return_type, CharT> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext&);

  template <class FormatContext>
  int format(format_function_invalid_return_type&, FormatContext&) const;
};
void test_format_function_invalid_return_type() {
  format_function_invalid_return_type t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization's format function does not return the required type.}}
  (void)std::make_format_args(t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization's format function does not return the required type.}}
  (void)std::make_wformat_args(t);
#endif
}

struct format_function_not_const_qualified {};
template <class CharT>
struct std::formatter<format_function_not_const_qualified, CharT> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext&);

  template <class FormatContext>
  typename FormatContext::iterator format(format_function_not_const_qualified&, FormatContext&);
};
void test_format_function_not_const_qualified() {
  format_function_not_const_qualified t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization's format function is not const qualified.}}
  (void)std::make_format_args(t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization's format function is not const qualified.}}
  (void)std::make_wformat_args(t);
#endif
}
