//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <format>

// template<class charT> struct runtime-format-string {  // exposition-only
// private:
//   basic_string_view<charT> str;  // exposition-only
//
// public:
//   runtime-format-string(basic_string_view<charT> s) noexcept : str(s) {}
//
//   runtime-format-string(const runtime-format-string&) = delete;
//   runtime-format-string& operator=(const runtime-format-string&) = delete;
// };
//
// runtime-format-string<char> runtime_format(string_view fmt) noexcept;
// runtime-format-string<wchar_t> runtime_format(wstring_view fmt) noexcept;
//
// Additional testing is done in
// - libcxx/test/std/utilities/format/format.functions/format.runtime_format.pass.cpp
// - libcxx/test/std/utilities/format/format.functions/format.locale.runtime_format.pass.cpp

#include <format>

#include <cassert>
#include <concepts>
#include <string_view>
#include <type_traits>

#include "test_macros.h"

template <class T, class CharT>
static void test_properties() {
  static_assert(std::is_nothrow_convertible_v<std::basic_string_view<CharT>, T>);
  static_assert(std::is_nothrow_constructible_v<T, std::basic_string_view<CharT>>);

  static_assert(!std::copy_constructible<T>);
  static_assert(!std::is_copy_assignable_v<T>);

  static_assert(!std::move_constructible<T>);
  static_assert(!std::is_move_assignable_v<T>);
}

int main(int, char**) {
  static_assert(noexcept(std::runtime_format(std::string_view{})));
  auto format_string = std::runtime_format(std::string_view{});

  using FormatString = decltype(format_string);
  LIBCPP_ASSERT((std::same_as<FormatString, std::__runtime_format_string<char>>));
  test_properties<FormatString, char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  static_assert(noexcept(std::runtime_format(std::wstring_view{})));
  auto wformat_string = std::runtime_format(std::wstring_view{});

  using WFormatString = decltype(wformat_string);
  LIBCPP_ASSERT((std::same_as<WFormatString, std::__runtime_format_string<wchar_t>>));
  test_properties<WFormatString, wchar_t>();
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
