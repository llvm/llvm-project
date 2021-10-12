//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// These tests require locale for non-char paths
// UNSUPPORTED: libcpp-has-no-localization

// <filesystem>

// class path

// std::string  string() const;
// std::wstring wstring() const;
// std::u8string  u8string() const;
// std::u16string u16string() const;
// std::u32string u32string() const;


#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "min_allocator.h"
#include "filesystem_test_helper.h"


MultiStringType longString = MKSTR("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/123456789/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");

int main(int, char**)
{
  using namespace fs;
  auto const& MS = longString;
  const char* value = longString;
  const path p(value);
  {
    std::string s = p.string();
    assert(s == value);
  }
  {
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    ASSERT_SAME_TYPE(decltype(p.u8string()), std::u8string);
    std::u8string s = p.u8string();
    assert(s == (const char8_t*)MS);
#else
    ASSERT_SAME_TYPE(decltype(p.u8string()), std::string);
    std::string s = p.u8string();
    assert(s == (const char*)MS);
#endif
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    std::wstring s = p.wstring();
    assert(s == (const wchar_t*)MS);
  }
#endif
  {
    std::u16string s = p.u16string();
    assert(s == (const char16_t*)MS);
  }
  {
    std::u32string s = p.u32string();
    assert(s == (const char32_t*)MS);
  }

  return 0;
}
