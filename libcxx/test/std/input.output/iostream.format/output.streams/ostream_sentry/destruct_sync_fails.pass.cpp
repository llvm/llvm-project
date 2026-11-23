//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ostream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ostream::sentry;

// ~sentry();
//
// LWG4188: If rdbuf()->pubsync() returns -1 or exits via an exception, sets badbit without propagating an exception.

// UNSUPPORTED: no-exceptions

// basic_ostream<char> and basic_ostream<wchar_t> are explicitly instantiated in the dylib, so the fixed
// sentry destructor is only used with a dylib that contains the fix.
// XFAIL: using-built-library-before-llvm-24

#include <cassert>
#include <ios>
#include <ostream>
#include <streambuf>

#include "test_macros.h"

template <class CharT>
struct failing_sync_buf : std::basic_streambuf<CharT> {
protected:
  virtual int sync() { return -1; }
};

template <class CharT>
struct throwing_sync_buf : std::basic_streambuf<CharT> {
protected:
  virtual int sync() { throw 1; }
};

template <class Buf, class CharT>
void test(std::ios_base::iostate except) {
  Buf sb;
  std::basic_ostream<CharT> os(&sb);
  os.exceptions(except);
  try {
    {
      typename std::basic_ostream<CharT>::sentry s(os);
      assert(bool(s));
      std::unitbuf(os);
    }
    assert(os.bad());
  } catch (...) {
    assert(false);
  }
}

template <class CharT>
void test() {
  test<failing_sync_buf<CharT>, CharT>(std::ios_base::goodbit);
  test<failing_sync_buf<CharT>, CharT>(std::ios_base::badbit);
  test<throwing_sync_buf<CharT>, CharT>(std::ios_base::goodbit);
  test<throwing_sync_buf<CharT>, CharT>(std::ios_base::badbit);
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
