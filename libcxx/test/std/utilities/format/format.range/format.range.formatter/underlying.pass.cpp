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

// constexpr formatter<T, charT>& underlying();
// constexpr const formatter<T, charT>& underlying() const;

#include <concepts>
#include <format>

#include "test_macros.h"

template <class CharT>
constexpr void test_underlying() {
  {
    std::range_formatter<int, CharT> formatter;
    [[maybe_unused]] std::same_as<std::formatter<int, CharT>&> decltype(auto) underlying = formatter.underlying();
  }
  {
    const std::range_formatter<int, CharT> formatter;
    [[maybe_unused]] std::same_as<const std::formatter<int, CharT>&> decltype(auto) underlying = formatter.underlying();
  }
}

constexpr bool test() {
  test_underlying<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_underlying<wchar_t>();
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
