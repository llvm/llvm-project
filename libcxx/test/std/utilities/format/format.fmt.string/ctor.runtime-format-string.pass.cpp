//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <format>

// template<class charT, class... Args>
// class basic_format_string<charT, type_identity_t<Args>...>
//
// basic_format_string(runtime-format-string<charT> s) noexcept : str(s.str) {}
//
// Additional testing is done in
// - libcxx/test/std/utilities/format/format.functions/format.runtime_format.pass.cpp
// - libcxx/test/std/utilities/format/format.functions/format.locale.runtime_format.pass.cpp

#include <format>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  static_assert(noexcept(std::format_string<>{std::runtime_format(std::string_view{})}));
  {
    std::format_string<> s = std::runtime_format("}{invalid format string}{");
    assert(s.get() == "}{invalid format string}{");
  }

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  static_assert(noexcept(std::wformat_string<>{std::runtime_format(std::wstring_view{})}));
  {
    std::wformat_string<> s = std::runtime_format(L"}{invalid format string}{");
    assert(s.get() == L"}{invalid format string}{");
  }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
