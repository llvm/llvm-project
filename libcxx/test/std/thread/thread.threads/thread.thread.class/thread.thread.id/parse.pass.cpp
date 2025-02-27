//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-threads

// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <thread>

// template<class charT>
// struct formatter<thread::id, charT>;

// template<class ParseContext>
//   constexpr typename ParseContext::iterator
//     parse(ParseContext& ctx);

// Note this tests the basics of this function. It's tested in more detail in
// the format functions test.

#include <cassert>
#include <concepts>
#include <format>
#include <memory>
#include <thread>

#include "test_macros.h"
#include "make_string.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class StringViewT>
constexpr void test_parse(StringViewT fmt, std::size_t offset) {
  using CharT    = typename StringViewT::value_type;
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::formatter<std::thread::id, CharT> formatter;
  static_assert(std::semiregular<decltype(formatter)>);

  std::same_as<typename StringViewT::iterator> auto it = formatter.parse(parse_ctx);
  // std::to_address works around LWG3989 and MSVC STL's iterator debugging mechanism.
  assert(std::to_address(it) == std::to_address(fmt.end()) - offset);
}

template <class CharT>
constexpr void test_fmt() {
  test_parse(SV(""), 0);
  test_parse(SV("1"), 0);

  test_parse(SV("}"), 1);
  test_parse(SV("1}"), 1);
}

constexpr bool test() {
  test_fmt<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_fmt<wchar_t>();
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
