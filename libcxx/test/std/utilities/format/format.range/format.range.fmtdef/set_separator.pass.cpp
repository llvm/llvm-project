//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <format>

// template<ranges::input_range R, class charT>
//   struct range-default-formatter<range_format::sequence, R, charT>

// constexpr void set_separator(basic_string_view<charT> sep) noexcept;

#include <format>
#include <cassert>
#include <iterator>
#include <type_traits>
#include <vector>

#include "make_string.h"
#include "test_format_context.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
constexpr void test_setter() {
  std::formatter<std::vector<int>, CharT> formatter;
  formatter.set_separator(SV("sep"));
  // Note the SV macro may throw, so can't use it.
  static_assert(noexcept(formatter.set_separator(std::basic_string_view<CharT>{})));

  // Note there is no direct way to validate this function modified the object.
  if (!std::is_constant_evaluated()) {
    using String     = std::basic_string<CharT>;
    using OutIt      = std::back_insert_iterator<String>;
    using FormatCtxT = std::basic_format_context<OutIt, CharT>;

    String result;
    OutIt out             = std::back_inserter(result);
    FormatCtxT format_ctx = test_format_context_create<OutIt, CharT>(out, std::make_format_args<FormatCtxT>());
    formatter.format(std::vector<int>{0, 42, 99}, format_ctx);
    assert(result == SV("[0sep42sep99]"));
  }
}

constexpr bool test() {
  test_setter<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_setter<wchar_t>();
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
