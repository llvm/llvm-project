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

// syncbuf_type* rdbuf() const noexcept

#include <syncstream>
#include <sstream>
#include <cassert>

#include "test_macros.h"

template <class CharT>
void test() {
  const std::basic_osyncstream<CharT> out{nullptr};
  assert(out.rdbuf() != nullptr);
  ASSERT_NOEXCEPT(out.rdbuf());
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
