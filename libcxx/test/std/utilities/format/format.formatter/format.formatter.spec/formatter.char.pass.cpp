//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// C++23 the formatter is a debug-enabled specialization.
// [format.formatter.spec]:
// Each header that declares the template `formatter` provides the following
// enabled specializations:
// The specializations
//   template<> struct formatter<char, char>;
//   template<> struct formatter<char, wchar_t>;
//   template<> struct formatter<wchar_t, wchar_t>;

#include <format>
#include <cassert>
#include <concepts>
#include <iterator>
#include <memory>
#include <type_traits>

#include "test_format_context.h"
#include "test_macros.h"
#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class StringT, class StringViewT, class ArgumentT>
void test(StringT expected, StringViewT fmt, ArgumentT arg, std::size_t offset) {
  using CharT = typename StringT::value_type;
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::formatter<ArgumentT, CharT> formatter;
  static_assert(std::semiregular<decltype(formatter)>);

  std::same_as<typename StringViewT::iterator> auto it = formatter.parse(parse_ctx);
  // std::to_address works around LWG3989 and MSVC STL's iterator debugging mechanism.
  assert(std::to_address(it) == std::to_address(fmt.end()) - offset);

  StringT result;
  auto out = std::back_inserter(result);
  using FormatCtxT = std::basic_format_context<decltype(out), CharT>;

  FormatCtxT format_ctx =
      test_format_context_create<decltype(out), CharT>(out, std::make_format_args<FormatCtxT>(arg));
  formatter.format(arg, format_ctx);
  assert(result == expected);
}

template <class StringT, class ArgumentT>
void test_termination_condition(StringT expected, StringT f, ArgumentT arg) {
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  using CharT = typename StringT::value_type;
  std::basic_string_view<CharT> fmt{f};
  assert(fmt.back() == CharT('}') && "Pre-condition failure");

  test(expected, fmt, arg, 1);
  fmt.remove_suffix(1);
  test(expected, fmt, arg, 0);
}

#if TEST_STD_VER > 20
template <class ArgumentT, class CharT>
constexpr bool test_set_debug_format() {
  std::formatter<ArgumentT, CharT> formatter;
  LIBCPP_ASSERT(formatter.__parser_.__type_ == std::__format_spec::__type::__default);

  formatter.set_debug_format();
  LIBCPP_ASSERT(formatter.__parser_.__type_ == std::__format_spec::__type::__debug);

  std::basic_string_view fmt = SV("d}");
  std::basic_format_parse_context<CharT> parse_ctx{fmt};
  formatter.parse(parse_ctx);
  LIBCPP_ASSERT(formatter.__parser_.__type_ == std::__format_spec::__type::__decimal);

  formatter.set_debug_format();
  LIBCPP_ASSERT(formatter.__parser_.__type_ == std::__format_spec::__type::__debug);

  return true;
}
#endif

template <class ArgumentT, class CharT>
void test_char_type() {
  test_termination_condition(STR("a"), STR("}"), ArgumentT('a'));
  test_termination_condition(STR("z"), STR("}"), ArgumentT('z'));
  test_termination_condition(STR("A"), STR("}"), ArgumentT('A'));
  test_termination_condition(STR("Z"), STR("}"), ArgumentT('Z'));
  test_termination_condition(STR("0"), STR("}"), ArgumentT('0'));
  test_termination_condition(STR("9"), STR("}"), ArgumentT('9'));

#if TEST_STD_VER > 20
  test_set_debug_format<ArgumentT, CharT>();
  static_assert(test_set_debug_format<ArgumentT, CharT>());
#endif
}

int main(int, char**) {
  test_char_type<char, char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_char_type<char, wchar_t>();
  test_char_type<wchar_t, wchar_t>();
#endif

  return 0;
}
