//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// basic_string(initializer_list<charT> il, const Allocator& a = Allocator()); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX20 bool test() {
  {
    std::string s = {'a', 'b', 'c'};
    assert(s == "abc");
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    std::wstring s;
    s = {L'a', L'b', L'c'};
    assert(s == L"abc");
  }
#endif
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    S s = {'a', 'b', 'c'};
    assert(s == "abc");
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    typedef std::basic_string<wchar_t, std::char_traits<wchar_t>, min_allocator<wchar_t>> S;
    S s;
    s = {L'a', L'b', L'c'};
    assert(s == L"abc");
  }
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
