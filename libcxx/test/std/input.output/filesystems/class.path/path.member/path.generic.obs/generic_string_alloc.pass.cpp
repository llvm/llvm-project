//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// These tests require locale for non-char paths
// UNSUPPORTED: no-localization

// <filesystem>

// class path

// template <class ECharT, class Traits = char_traits<ECharT>,
//           class Allocator = allocator<ECharT>>
// basic_string<ECharT, Traits, Allocator>
// generic_string(const Allocator& a = Allocator()) const;

#include <filesystem>
#include <cassert>
#include <string>
#include <type_traits>

#include "count_new.h"
#include "make_string.h"
#include "min_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"
namespace fs = std::filesystem;

MultiStringType longString = MKSTR("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/123456789/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");

// generic_string<C, T, A> forwards to string<C, T, A>. Tests for
// string<C, T, A>() are in "path.native.obs/string_alloc.pass.cpp".
// generic_string is minimally tested here.
template <class CharT>
void doAllocTest()
{
  using namespace fs;
  using Traits = std::char_traits<CharT>;
  using Alloc = malloc_allocator<CharT>;
  using Str = std::basic_string<CharT, Traits, Alloc>;
  const CharT* expect = longString;
  const path p((const char*)longString);
  {
    // On Windows, charset conversions cause allocations outside of the
    // provided allocator.
    TEST_NOT_WIN32(DisableAllocationGuard g);
    Alloc a;
    Alloc::disable_default_constructor = true;
    Str s = p.generic_string<CharT, Traits, Alloc>(a);
    assert(s == expect);
    assert(Alloc::alloc_count > 0);
    assert(Alloc::outstanding_alloc() == 1);
    Alloc::disable_default_constructor = false;
  }
}

int main(int, char**)
{
  doAllocTest<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  doAllocTest<wchar_t>();
#endif
  doAllocTest<char16_t>();
  doAllocTest<char32_t>();
#if TEST_STD_VER > 17 && defined(__cpp_lib_char8_t)
  doAllocTest<char8_t>();
#endif
  return 0;
}
