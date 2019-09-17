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

// template <class ECharT, class Traits = char_traits<ECharT>,
//           class Allocator = allocator<ECharT>>
// basic_string<ECharT, Traits, Allocator>
// generic_string(const Allocator& a = Allocator()) const;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "min_allocator.h"
#include "filesystem_test_helper.h"

MultiStringType longString = MKSTR("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/123456789/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");


// generic_string<C, T, A> forwards to string<C, T, A>. Tests for
// string<C, T, A>() are in "path.native.op/string_alloc.pass.cpp".
// generic_string is minimally tested here.
int main(int, char**)
{
  using namespace fs;
  using CharT = wchar_t;
  using Traits = std::char_traits<CharT>;
  using Alloc = malloc_allocator<CharT>;
  using Str = std::basic_string<CharT, Traits, Alloc>;
  const wchar_t* expect = longString;
  const path p((const char*)longString);
  {
    DisableAllocationGuard g;
    Alloc a;
    Alloc::disable_default_constructor = true;
    Str s = p.generic_string<wchar_t, Traits, Alloc>(a);
    assert(s == expect);
    assert(Alloc::alloc_count > 0);
    assert(Alloc::outstanding_alloc() == 1);
  }

  return 0;
}
