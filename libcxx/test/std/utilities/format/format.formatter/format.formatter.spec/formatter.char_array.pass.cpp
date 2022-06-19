//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format
// TODO FMT Evaluate gcc-11 status
// UNSUPPORTED: gcc-11

// <format>

// [format.functions]/25
//   Preconditions: formatter<remove_cvref_t<Ti>, charT> meets the
//   BasicFormatter requirements ([formatter.requirements]) for each Ti in
//   Args.

#include <format>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "test_macros.h"
#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)
#define CSTR(S) MAKE_CSTRING(CharT, S)

// This is based on the method found in
// clang/test/CXX/temp/temp.arg/temp.arg.nontype/p1-cxx20.cpp
template <size_t N>
struct Tester {
  // This is not part of the real test, but is used the deduce the size of the input.
  constexpr Tester(const char (&r)[N]) { __builtin_memcpy(text, r, N); }
  char text[N];

  // The size of the array shouldn't include the NUL character.
  static const size_t size = N - 1;

  template <class CharT>
  void test(const std::basic_string<CharT>& expected, const std::basic_string_view<CharT>& fmt) const {
    using Str = CharT[size];
    std::basic_format_parse_context<CharT> parse_ctx{fmt};
    std::formatter<Str, CharT> formatter;
    static_assert(std::semiregular<decltype(formatter)>);

    auto it = formatter.parse(parse_ctx);
    assert(it == fmt.end() - (!fmt.empty() && fmt.back() == '}'));

    std::basic_string<CharT> result;
    auto out = std::back_inserter(result);
    using FormatCtxT = std::basic_format_context<decltype(out), CharT>;

    std::basic_string<CharT> buffer{text, text + N};
    // Note not too found of this hack
    Str* data = reinterpret_cast<Str*>(const_cast<CharT*>(buffer.c_str()));

    auto format_ctx = std::__format_context_create<decltype(out), CharT>(out, std::make_format_args<FormatCtxT>(*data));
    formatter.format(*data, format_ctx);
    assert(result == expected);
  }

  template <class CharT>
  void test_termination_condition(const std::basic_string<CharT>& expected, const std::basic_string<CharT>& f) const {
    // The format-spec is valid if completely consumed or terminates at a '}'.
    // The valid inputs all end with a '}'. The test is executed twice:
    // - first with the terminating '}',
    // - second consuming the entire input.
    std::basic_string_view<CharT> fmt{f};
    assert(fmt.back() == CharT('}') && "Pre-condition failure");

    test(expected, fmt);
    fmt.remove_suffix(1);
    test(expected, fmt);
  }
};

template <Tester t, class CharT>
void test_helper_wrapper(std::basic_string<CharT> expected, std::basic_string<CharT> fmt) {
  t.test_termination_condition(expected, fmt);
}

template <class CharT>
void test_array() {
  test_helper_wrapper<" azAZ09,./<>?">(STR(" azAZ09,./<>?"), STR("}"));

  std::basic_string<CharT> s(CSTR("abc\0abc"), 7);
  test_helper_wrapper<"abc\0abc">(s, STR("}"));

  test_helper_wrapper<"world">(STR("world"), STR("}"));
  test_helper_wrapper<"world">(STR("world"), STR("_>}"));

  test_helper_wrapper<"world">(STR("   world"), STR(">8}"));
  test_helper_wrapper<"world">(STR("___world"), STR("_>8}"));
  test_helper_wrapper<"world">(STR("_world__"), STR("_^8}"));
  test_helper_wrapper<"world">(STR("world___"), STR("_<8}"));

  test_helper_wrapper<"world">(STR("world"), STR(".5}"));
  test_helper_wrapper<"universe">(STR("unive"), STR(".5}"));

  test_helper_wrapper<"world">(STR("%world%"), STR("%^7.7}"));
  test_helper_wrapper<"universe">(STR("univers"), STR("%^7.7}"));
}

int main(int, char**) {
  test_array<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_array<wchar_t>();
#endif

  return 0;
}
