//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-experimental-syncstream

// <syncstream>

// template <class charT, class traits, class Allocator>
// class basic_osyncstream;

// void emit();

#include <syncstream>
#include <sstream>
#include <cassert>

#include "test_macros.h"

template <class CharT>
void test() {
  using OS = std::basic_osyncstream<CharT>;
  using SS = std::basic_ostringstream<CharT>;
  CharT c  = 'f';

  SS ss;
  OS out(ss);
  out << c;
  assert(ss.str().empty());
  out.emit();
  assert(ss.str()[0] == c);
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
}
