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

// basic_ostringstream(ios_base::openmode which, const Allocator& a);

#include <sstream>
#include <cassert>

#include "test_allocator.h"
#include "test_macros.h"

template <class CharT>
static void test() {
  const test_allocator<CharT> a(2);
  const std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(std::ios_base::binary, a);
  assert(ss.rdbuf()->get_allocator() == a);
  assert(ss.view().empty());
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  return 0;
}
