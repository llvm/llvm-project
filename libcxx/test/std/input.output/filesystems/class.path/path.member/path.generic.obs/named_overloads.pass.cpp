//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// These tests require locale for non-char paths
// UNSUPPORTED: no-localization

// <filesystem>

// class path

// std::string  generic_string() const;
// std::wstring generic_wstring() const;
// std::u8string  generic_u8string() const;
// std::u16string generic_u16string() const;
// std::u32string generic_u32string() const;


#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "min_allocator.h"
#include "filesystem_test_helper.h"

MultiStringType input = MKSTR("c:\\foo\\bar");
#ifdef _WIN32
// On windows, the generic_* accessors return a path with forward slashes
MultiStringType ref = MKSTR("c:/foo/bar");
#else
// On posix, the input string is returned as-is
MultiStringType ref = MKSTR("c:\\foo\\bar");
#endif

int main(int, char**)
{
  using namespace fs;
  auto const& MS = ref;
  const char* value = input;
  const path p(value);
  {
    std::string s = p.generic_string();
    assert(s == (const char*)MS);
  }
  {
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    ASSERT_SAME_TYPE(decltype(p.generic_u8string()), std::u8string);
    std::u8string s = p.generic_u8string();
    assert(s == (const char8_t*)MS);
#else
    ASSERT_SAME_TYPE(decltype(p.generic_u8string()), std::string);
    std::string s = p.generic_u8string();
    assert(s == (const char*)MS);
#endif
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    std::wstring s = p.generic_wstring();
    assert(s == (const wchar_t*)MS);
  }
#endif
  {
    std::u16string s = p.generic_u16string();
    assert(s == (const char16_t*)MS);
  }
  {
    std::u32string s = p.generic_u32string();
    assert(s == (const char32_t*)MS);
  }

  return 0;
}
