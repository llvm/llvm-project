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
// class basic_istringstream

// basic_istringstream(ios_base::openmode which, const Allocator& a);

#include <sstream>
#include <cassert>

#include "test_allocator.h"
#include "test_macros.h"
#include "operator_hijacker.h"

template <class CharT, class Allocator>
static void test(const Allocator& a) {
  const std::basic_istringstream<CharT, std::char_traits<CharT>, Allocator> ss(std::ios_base::binary, a);
  assert(ss.rdbuf()->get_allocator() == a);
  assert(ss.view().empty());
}

int main(int, char**) {
  test<char>(test_allocator<char>(2));
  test<char>(operator_hijacker_allocator<char>());
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>(test_allocator<wchar_t>(2));
  test<wchar_t>(operator_hijacker_allocator<wchar_t>());
#endif

  return 0;
}
