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

// template <class ECharT, class Traits = char_traits<ECharT>,
//           class Allocator = allocator<ECharT>>
// basic_string<ECharT, Traits, Allocator>
// string(const Allocator& a = Allocator()) const;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "min_allocator.h"
#include "filesystem_test_helper.h"


// the SSO is always triggered for strings of size 2.
MultiStringType shortString = MKSTR("a");
MultiStringType longString = MKSTR("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/123456789/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");

template <class CharT>
void doShortStringTest(MultiStringType const& MS) {
  using namespace fs;
  using Ptr = CharT const*;
  using Str = std::basic_string<CharT>;
  using Alloc = std::allocator<CharT>;
  Ptr value = MS;
  const path p((const char*)MS);
#ifdef _WIN32
  // On Windows, charset conversions cause allocations outside of the
  // provided allocator, but accessing the native type should work without
  // extra allocations.
  bool DisableAllocations = std::is_same<CharT, path::value_type>::value;
#else
  // On other platforms, these methods only use the provided allocator, and
  // no extra allocations should be done.
  bool DisableAllocations = true;
#endif
  {
      DisableAllocationGuard g(DisableAllocations);
      Str s = p.string<CharT>();
      assert(s == value);
      Str s2 = p.string<CharT>(Alloc{});
      assert(s2 == value);
  }
  using MAlloc = malloc_allocator<CharT>;
  MAlloc::reset();
  {
      using Traits = std::char_traits<CharT>;
      using AStr = std::basic_string<CharT, Traits, MAlloc>;
      DisableAllocationGuard g(DisableAllocations);
      AStr s = p.string<CharT, Traits, MAlloc>();
      assert(s == value);
      assert(MAlloc::alloc_count == 0);
      assert(MAlloc::outstanding_alloc() == 0);
  }
  MAlloc::reset();
  { // Other allocator - provided copy
      using Traits = std::char_traits<CharT>;
      using AStr = std::basic_string<CharT, Traits, MAlloc>;
      DisableAllocationGuard g(DisableAllocations);
      MAlloc a;
      // don't allow another allocator to be default constructed.
      MAlloc::disable_default_constructor = true;
      AStr s = p.string<CharT, Traits, MAlloc>(a);
      assert(s == value);
      assert(MAlloc::alloc_count == 0);
      assert(MAlloc::outstanding_alloc() == 0);
  }
  MAlloc::reset();
}

template <class CharT>
void doLongStringTest(MultiStringType const& MS) {
  using namespace fs;
  using Ptr = CharT const*;
  using Str = std::basic_string<CharT>;
  Ptr value = MS;
  const path p((const char*)MS);
  { // Default allocator
      using Alloc = std::allocator<CharT>;
      Str s = p.string<CharT>();
      assert(s == value);
      Str s2 = p.string<CharT>(Alloc{});
      assert(s2 == value);
  }
  using MAlloc = malloc_allocator<CharT>;
  MAlloc::reset();
#ifdef _WIN32
  // On Windows, charset conversions cause allocations outside of the
  // provided allocator, but accessing the native type should work without
  // extra allocations.
  bool DisableAllocations = std::is_same<CharT, path::value_type>::value;
#else
  // On other platforms, these methods only use the provided allocator, and
  // no extra allocations should be done.
  bool DisableAllocations = true;
#endif

  { // Other allocator - default construct
      using Traits = std::char_traits<CharT>;
      using AStr = std::basic_string<CharT, Traits, MAlloc>;
      DisableAllocationGuard g(DisableAllocations);
      AStr s = p.string<CharT, Traits, MAlloc>();
      assert(s == value);
      assert(MAlloc::alloc_count > 0);
      assert(MAlloc::outstanding_alloc() == 1);
  }
  MAlloc::reset();
  { // Other allocator - provided copy
      using Traits = std::char_traits<CharT>;
      using AStr = std::basic_string<CharT, Traits, MAlloc>;
      DisableAllocationGuard g(DisableAllocations);
      MAlloc a;
      // don't allow another allocator to be default constructed.
      MAlloc::disable_default_constructor = true;
      AStr s = p.string<CharT, Traits, MAlloc>(a);
      assert(s == value);
      assert(MAlloc::alloc_count > 0);
      assert(MAlloc::outstanding_alloc() == 1);
  }
  MAlloc::reset();
  /////////////////////////////////////////////////////////////////////////////
}

int main(int, char**)
{
  using namespace fs;
  {
    auto const& S = shortString;
    doShortStringTest<char>(S);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    doShortStringTest<wchar_t>(S);
#endif
    doShortStringTest<char16_t>(S);
    doShortStringTest<char32_t>(S);
#if TEST_STD_VER > 17 && defined(__cpp_lib_char8_t)
    doShortStringTest<char8_t>(S);
#endif
  }
  {
    auto const& S = longString;
    doLongStringTest<char>(S);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    doLongStringTest<wchar_t>(S);
#endif
    doLongStringTest<char16_t>(S);
    doLongStringTest<char32_t>(S);
#if TEST_STD_VER > 17 && defined(__cpp_lib_char8_t)
    doLongStringTest<char8_t>(S);
#endif
  }

  return 0;
}
