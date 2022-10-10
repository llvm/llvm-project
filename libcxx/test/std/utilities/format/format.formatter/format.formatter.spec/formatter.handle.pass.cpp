//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// <format>

// A user defined formatter using
// template<class Context>
// class basic_format_arg<Context>::handle

#include <format>

#include <array>
#include <cassert>
#include <cmath>
#include <charconv>
#include <concepts>
#include <string>
#include <type_traits>

#include "test_format_context.h"
#include "test_macros.h"

enum class color { black, red, gold };
const char* color_names[] = {"black", "red", "gold"};

template <>
struct std::formatter<color> : std::formatter<const char*> {
  auto format(color c, auto& ctx) const {
    return formatter<const char*>::format(color_names[static_cast<int>(c)], ctx);
  }
};

void test(std::string expected, std::string_view fmt, color arg) {
  auto parse_ctx = std::format_parse_context(fmt);
  std::formatter<color, char> formatter;
  static_assert(std::semiregular<decltype(formatter)>);

  auto it = formatter.parse(parse_ctx);
  assert(it == fmt.end() - (!fmt.empty() && fmt.back() == '}'));

  std::string result;
  auto out = std::back_inserter(result);
  using FormatCtxT = std::basic_format_context<decltype(out), char>;

  FormatCtxT format_ctx =
      test_format_context_create<decltype(out), char>(out, std::make_format_args<FormatCtxT>(arg));
  formatter.format(arg, format_ctx);
  assert(result == expected);
}

void test_termination_condition(std::string expected, std::string f, color arg) {
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  std::string_view fmt{f};
  assert(fmt.back() == '}' && "Pre-condition failure");

  test(expected, fmt, arg);
  fmt.remove_suffix(1);
  test(expected, fmt, arg);
}

int main(int, char**) {
  test_termination_condition("black", "}", color::black);
  test_termination_condition("red", "}", color::red);
  test_termination_condition("gold", "}", color::gold);

  return 0;
}
