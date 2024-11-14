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

// explicit basic_ostringstream(basic_string<char_type, traits_type, allocator_type>&& s, ios_base::openmode which = ios_base::out);

#include <sstream>
#include <cassert>

#include "make_string.h"
#include "test_allocator.h"
#include "test_macros.h"
#include "operator_hijacker.h"

#define STR(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT, class Allocator>
static void test() {
  {
    std::basic_string<CharT, std::char_traits<CharT>, Allocator> s(STR("testing"));
    const std::basic_ostringstream<CharT, std::char_traits<CharT>, Allocator> ss(std::move(s));
    assert(ss.str() == SV("testing"));
  }
  {
    std::basic_string<CharT, std::char_traits<CharT>, Allocator> s(STR("testing"));
    const std::basic_ostringstream<CharT, std::char_traits<CharT>, Allocator> ss(std::move(s), std::ios_base::binary);
    assert(ss.str() == SV("testing"));
  }
}

int main(int, char**) {
  test<char, std::allocator<char>>();
  test<char, test_allocator<char>>();
  test<char, operator_hijacker_allocator<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t, std::allocator<wchar_t>>();
  test<wchar_t, test_allocator<wchar_t>>();
  test<wchar_t, operator_hijacker_allocator<wchar_t>>();
#endif
  return 0;
}
