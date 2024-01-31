//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT>>
// class basic_stringbuf

// template<class T>
//   void str(const T& t);

#include <array>
#include <cassert>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

#include "test_macros.h"

using namespace std::string_literals;
using namespace std::string_view_literals;

template <typename S, typename T>
concept HasStr = requires(S s, const T sv) {
  { s.str(sv) };
};

struct SomeObject {};

// std::stringbuf
static_assert(HasStr<std::stringbuf, std::string_view>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!HasStr<std::stringbuf, std::wstring_view>);
#endif
static_assert(HasStr<std::stringbuf, const char*>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!HasStr<std::stringbuf, const wchar_t*>);
#endif
static_assert(HasStr<std::stringbuf, std::string>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!HasStr<std::stringbuf, std::wstring>);
#endif
static_assert(!HasStr<std::stringbuf, std::array<char, 0>>);
static_assert(!HasStr<std::stringbuf, char>);
static_assert(!HasStr<std::stringbuf, int>);
static_assert(!HasStr<std::stringbuf, SomeObject>);
static_assert(!HasStr<std::stringbuf, std::nullptr_t>);

// std::wstringbuf

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(HasStr<std::wstringbuf, std::wstring_view>);
static_assert(!HasStr<std::wstringbuf, std::string_view>);
static_assert(HasStr<std::wstringbuf, const wchar_t*>);
static_assert(!HasStr<std::wstringbuf, const char*>);
static_assert(HasStr<std::wstringbuf, std::wstring>);
static_assert(!HasStr<std::wstringbuf, std::string>);
static_assert(!HasStr<std::stringbuf, std::array<wchar_t, 0>>);
static_assert(!HasStr<std::wstringbuf, wchar_t>);
static_assert(!HasStr<std::wstringbuf, int>);
static_assert(!HasStr<std::wstringbuf, SomeObject>);
static_assert(!HasStr<std::wstringbuf, std::nullptr_t>);
#endif

int main(int, char**) {
  {
    std::stringbuf ss;
    assert(ss.str().empty());
    ss.str("ba");
    assert(ss.str() == "ba");
    ss.str("ma"sv);
    assert(ss.str() == "ma");
    ss.str("zmt"s);
    assert(ss.str() == "zmt");
    const std::string s;
    ss.str(s);
    assert(ss.str().empty());
  }

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    std::wstringbuf ss;
    assert(ss.str().empty());
    ss.str(L"ba");
    assert(ss.str() == L"ba");
    ss.str(L"ma"sv);
    assert(ss.str() == L"ma");
    ss.str(L"zmt"s);
    assert(ss.str() == L"zmt");
    const std::wstring s;
    ss.str(s);
    assert(ss.str().empty());
  }
#endif

  return 0;
}
