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
// class basic_istringstream

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

// std::istringstream
static_assert(HasStr<std::istringstream, std::string_view>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!HasStr<std::istringstream, std::wstring_view>);
#endif
static_assert(HasStr<std::istringstream, const char*>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!HasStr<std::istringstream, const wchar_t*>);
#endif
static_assert(HasStr<std::istringstream, std::string>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!HasStr<std::istringstream, std::wstring>);
#endif
static_assert(!HasStr<std::istringstream, std::array<char, 0>>);
static_assert(!HasStr<std::istringstream, char>);
static_assert(!HasStr<std::istringstream, int>);
static_assert(!HasStr<std::istringstream, SomeObject>);
static_assert(!HasStr<std::istringstream, std::nullptr_t>);

// std::wistringstream

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(HasStr<std::wistringstream, std::wstring_view>);
static_assert(!HasStr<std::wistringstream, std::string_view>);
static_assert(HasStr<std::wistringstream, const wchar_t*>);
static_assert(!HasStr<std::wistringstream, const char*>);
static_assert(HasStr<std::wistringstream, std::wstring>);
static_assert(!HasStr<std::wistringstream, std::string>);
static_assert(!HasStr<std::istringstream, std::array<wchar_t, 0>>);
static_assert(!HasStr<std::wistringstream, wchar_t>);
static_assert(!HasStr<std::wistringstream, int>);
static_assert(!HasStr<std::wistringstream, SomeObject>);
static_assert(!HasStr<std::wistringstream, std::nullptr_t>);
#endif

int main(int, char**) {
  {
    std::istringstream ss;
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
    std::wistringstream ss;
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
