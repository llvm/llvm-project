//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <filesystem>

// class path

// std::string  string() const;
// std::wstring wstring() const;
// std::u8string  u8string() const;
// std::u16string u16string() const;
// std::u32string u32string() const;


#include "filesystem_include.hpp"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.hpp"
#include "min_allocator.h"
#include "filesystem_test_helper.hpp"


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
    std::string s = p.u8string();
    assert(s == (const char*)MS);
  }
  {
    std::wstring s = p.wstring();
    assert(s == (const wchar_t*)MS);
  }
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
