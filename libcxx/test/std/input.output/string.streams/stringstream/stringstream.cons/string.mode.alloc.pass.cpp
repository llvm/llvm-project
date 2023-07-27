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
// class basic_stringstream

// template <class SAlloc>
// basic_stringstream(const basic_string<char_type, traits_type, SAlloc>& s, ios_base::openmode which, const Allocator& a)

#include <sstream>
#include <cassert>

#include "make_string.h"
#include "test_allocator.h"
#include "test_macros.h"

#define STR(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static void test() {
  const std::basic_string<CharT> s(STR("testing"));
  const test_allocator<CharT> a(2);
  const std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(s, std::ios_base::out, a);
  assert(ss.rdbuf()->get_allocator() == a);
  assert(ss.view() == SV("testing"));
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  return 0;
}
