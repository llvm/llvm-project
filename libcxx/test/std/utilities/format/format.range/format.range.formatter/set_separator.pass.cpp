//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT Fix this test using GCC, it currently times out.
// UNSUPPORTED: gcc-12

// This test requires the dylib support introduced in D92214.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{.+}}
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx11.{{.+}}

// <format>

// template<class T, class charT = char>
//   requires same_as<remove_cvref_t<T>, T> && formattable<T, charT>
// class range_formatter

// constexpr void set_separator(basic_string_view<charT> sep);

// Note this tests the basics of this function. It's tested in more detail in
// the format functions test.

#include <format>
#include <cassert>
#include <type_traits>
#include <vector>

#include "make_string.h"
#include "test_format_context.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
constexpr void test_setter() {
  std::range_formatter<int, CharT> formatter;
  formatter.set_separator(SV("sep"));

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
