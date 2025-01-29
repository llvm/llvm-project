//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <format>

// [format.formatter.spec]:
// Each header that declares the template `formatter` provides the following
// enabled specializations:
// ...
// For each charT, the pointer type specializations
// - template<> struct formatter<nullptr_t, charT>;
// - template<> struct formatter<void*, charT>;
// - template<> struct formatter<const void*, charT>;

#include <format>

#include <array>
#include <cassert>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>

#include "test_format_context.h"
#include "test_macros.h"
#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class StringT, class StringViewT, class PointerT>
void test(StringT expected, StringViewT fmt, PointerT arg, std::size_t offset) {
  using CharT = typename StringT::value_type;
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::formatter<PointerT, CharT> formatter;
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

  if (expected.empty()) {
    std::array<char, 128> buffer;
    buffer[0] = CharT('0');
    buffer[1] = CharT('x');
    expected.append(buffer.data(),
                    std::to_chars(buffer.data() + 2, buffer.data() + buffer.size(), reinterpret_cast<std::uintptr_t>(arg), 16).ptr);
  }
  assert(result == expected);
}

template <class StringT, class PointerT>
void test_termination_condition(StringT expected, StringT f, PointerT arg) {
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

template <class CharT>
void test_nullptr_t() {
  test_termination_condition(STR("0x0"), STR("}"), nullptr);
}

template <class PointerT, class CharT>
void test_pointer_type() {
  test_termination_condition(STR("0x0"), STR("}"), PointerT(0));
  test_termination_condition(STR("0x42"), STR("}"), PointerT(0x42));
  test_termination_condition(STR("0xffff"), STR("}"), PointerT(0xffff));
  test_termination_condition(STR(""), STR("}"), PointerT(-1));
}

template <class CharT>
void test_all_pointer_types() {
  test_nullptr_t<CharT>();
  test_pointer_type<void*, CharT>();
  test_pointer_type<const void*, CharT>();
}

int main(int, char**) {
  test_all_pointer_types<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_all_pointer_types<wchar_t>();
#endif

  return 0;
}
