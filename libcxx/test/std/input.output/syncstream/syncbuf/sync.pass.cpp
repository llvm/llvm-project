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

// protected:
// [syncstream.syncbuf.virtuals], overridden virtual functions
// int sync() override;

#include <syncstream>
#include <sstream>
#include <ostream>
#include <cassert>

#include "test_macros.h"

template <class CharT>
void test_sync(bool emit_on_sync) {
  std::basic_stringbuf<CharT> base;
  std::basic_syncbuf<CharT> buff(&base);
  std::basic_ostream<CharT> out(&buff);

  buff.set_emit_on_sync(emit_on_sync);

  out << 'a';
  out.flush(); // This is an indirect call to sync.

  if (emit_on_sync) {
    assert(base.str().size() == 1);
    assert(base.str()[0] == CharT('a'));
  } else
    assert(base.str().empty());
}

int main(int, char**) {
  test_sync<char>(true);
  test_sync<char>(false);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_sync<wchar_t>(true);
  test_sync<wchar_t>(false);
#endif

  return 0;
}
