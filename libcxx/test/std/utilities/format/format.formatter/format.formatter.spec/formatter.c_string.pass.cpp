//===----------------------------------------------------------------------===//
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
// For each `charT`, the string type specializations
//   template<> struct formatter<charT*, charT>;
//   template<> struct formatter<const charT*, charT>;

#include <format>
#include <cassert>
#include <concepts>
#include <iterator>
#include <type_traits>

#include "test_format_context.h"
#include "test_macros.h"
#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)
#define CSTR(S) MAKE_CSTRING(CharT, S)

template <class T, class StringT, class StringViewT, class CharT>
void test(StringT expected, StringViewT fmt, const CharT* a, std::size_t offset) {
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::formatter<T, CharT> formatter;
  static_assert(std::semiregular<decltype(formatter)>);

  auto it = formatter.parse(parse_ctx);
  assert(it == fmt.end() - offset);

  StringT result;
  auto out = std::back_inserter(result);
  using FormatCtxT = std::basic_format_context<decltype(out), CharT>;

  auto* arg = const_cast<T>(a);
  FormatCtxT format_ctx =
      test_format_context_create<decltype(out), CharT>(out, std::make_format_args<FormatCtxT>(arg));
  formatter.format(arg, format_ctx);
  assert(result == expected);
}

template <class ArgumentT, class StringT, class CharT>
void test_termination_condition(StringT expected, StringT f, const CharT* arg) {
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  std::basic_string_view<CharT> fmt{f};
  assert(fmt.back() == CharT('}') && "Pre-condition failure");

  test<ArgumentT>(expected, fmt, arg, 1);
  fmt.remove_suffix(1);
  test<ArgumentT>(expected, fmt, arg, 0);
}

#if TEST_STD_VER > 20
template <class ArgumentT, class CharT>
constexpr bool test_set_debug_format() {
  std::formatter<ArgumentT, CharT> formatter;
  LIBCPP_ASSERT(formatter.__parser_.__type_ == std::__format_spec::__type::__default);

  formatter.set_debug_format();
  LIBCPP_ASSERT(formatter.__parser_.__type_ == std::__format_spec::__type::__debug);

  std::basic_string_view fmt = SV("s}");
  std::basic_format_parse_context<CharT> parse_ctx{fmt};
  formatter.parse(parse_ctx);
  LIBCPP_ASSERT(formatter.__parser_.__type_ == std::__format_spec::__type::__string);

  formatter.set_debug_format();
  LIBCPP_ASSERT(formatter.__parser_.__type_ == std::__format_spec::__type::__debug);

  return true;
}
#endif

template <class ArgumentT>
void test_char_pointer() {
  using CharT = std::remove_cv_t<std::remove_pointer_t<ArgumentT>>;

  test_termination_condition<ArgumentT>(STR(" azAZ09,./<>?"), STR("}"),
                                        CSTR(" azAZ09,./<>?"));

  std::basic_string<CharT> s(CSTR("abc\0abc"), 7);
  test_termination_condition<ArgumentT>(STR("abc"), STR("}"), s.c_str());

  test_termination_condition<ArgumentT>(STR("world"), STR("}"), CSTR("world"));
  test_termination_condition<ArgumentT>(STR("world"), STR("_>}"),
                                        CSTR("world"));

  test_termination_condition<ArgumentT>(STR("   world"), STR(">8}"),
                                        CSTR("world"));
  test_termination_condition<ArgumentT>(STR("___world"), STR("_>8}"),
                                        CSTR("world"));
  test_termination_condition<ArgumentT>(STR("_world__"), STR("_^8}"),
                                        CSTR("world"));
  test_termination_condition<ArgumentT>(STR("world___"), STR("_<8}"),
                                        CSTR("world"));

  test_termination_condition<ArgumentT>(STR("world"), STR(".5}"),
                                        CSTR("world"));
  test_termination_condition<ArgumentT>(STR("unive"), STR(".5}"),
                                        CSTR("universe"));

  test_termination_condition<ArgumentT>(STR("%world%"), STR("%^7.7}"),
                                        CSTR("world"));
  test_termination_condition<ArgumentT>(STR("univers"), STR("%^7.7}"),
                                        CSTR("universe"));

#if TEST_STD_VER > 20
  test_set_debug_format<ArgumentT, CharT>();
  static_assert(test_set_debug_format<ArgumentT, CharT>());
#endif
}

int main(int, char**) {
  test_char_pointer<char*>();
  test_char_pointer<const char*>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_char_pointer<wchar_t*>();
  test_char_pointer<const wchar_t*>();
#endif

  return 0;
}
