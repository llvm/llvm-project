//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// libc++ supports basic_format_string in C++20 as an extension
// UNSUPPORTED: !stdlib=libc++ && c++20

// <format>

// template<class charT, class... Args>
// class basic_format_string<charT, type_identity_t<Args>...>
//
// constexpr basic_string_view<charT> get() const noexcept { return str; }

#include <format>

#include <cassert>
#include <concepts>
#include <string_view>

#include "test_macros.h"
#include "make_string.h"

#define CSTR(S) MAKE_CSTRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
constexpr bool test() {
  assert((std::basic_format_string<CharT>{CSTR("foo")}.get() == SV("foo")));
  assert((std::basic_format_string<CharT, int>{CSTR("{}")}.get() == SV("{}")));
  assert((std::basic_format_string<CharT, int, float>{CSTR("{} {:01.23L}")}.get() == SV("{} {:01.23L}")));

  // Embedded NUL character
  assert((std::basic_format_string<CharT, void*, double>{SV("{}\0{}")}.get() == SV("{}\0{}")));
  return true;
}

int main(int, char**) {
  test<char>();
  static_assert(test<char>());
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
  static_assert(test<wchar_t>());
#endif
  return 0;
}
