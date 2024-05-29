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
// class basic_syncbuf;

// allocator_type get_allocator() const noexcept;

#include <syncstream>
#include <cassert>

#include "test_macros.h"
#include "../helpers.h"

template <class T>
void test_get_allocator() {
  test_buf<T> base;
  test_allocator<T> alloc(42);
  const test_syncbuf<T, test_allocator<T>> buff(&base, alloc);
  assert(buff.get_allocator().id == 42);
  ASSERT_NOEXCEPT(buff.get_allocator());
}

int main(int, char**) {
  test_get_allocator<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_get_allocator<wchar_t>();
#endif

  return 0;
}
