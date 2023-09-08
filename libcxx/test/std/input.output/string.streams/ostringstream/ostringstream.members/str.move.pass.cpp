//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT>>
// class basic_ostringstream

// basic_string<charT, traits, Allocator> str() &&;

#include <sstream>
#include <cassert>

#include "make_string.h"
#include "test_macros.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class CharT>
static void test() {
  {
    std::basic_ostringstream<CharT> ss(STR("testing"));
    std::basic_string<CharT> s = std::move(ss).str();
    assert(s == STR("testing"));
    assert(ss.view().empty());
  }
  {
    std::basic_ostringstream<CharT> ss;
    std::basic_string<CharT> s = std::move(ss).str();
    assert(s.empty());
    assert(ss.view().empty());
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  return 0;
}
