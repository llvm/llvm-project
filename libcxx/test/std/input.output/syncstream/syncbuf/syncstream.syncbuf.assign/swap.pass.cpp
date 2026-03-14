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

// void swap(basic_syncbuf& other) noexcept;

#include <syncstream>
#include <sstream>
#include <cassert>

#include "test_macros.h"

#include <iostream>

template <class CharT>
static void test_basic() {
  std::basic_stringbuf<CharT> sstr1;
  std::basic_stringbuf<CharT> sstr2;
  std::basic_string<CharT> expected(42, CharT('*')); // a long string

  {
    std::basic_syncbuf<CharT> sync_buf1(&sstr1);
    sync_buf1.sputc(CharT('A')); // a short string

    std::basic_syncbuf<CharT> sync_buf2(&sstr2);
    sync_buf2.sputn(expected.data(), expected.size());

#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
    assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&sstr1) == 1);
    assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&sstr2) == 1);
#endif

    sync_buf1.swap(sync_buf2);
    assert(sync_buf1.get_wrapped() == &sstr2);
    assert(sync_buf2.get_wrapped() == &sstr1);

    assert(sstr1.str().empty());
    assert(sstr2.str().empty());

#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
    assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&sstr1) == 1);
    assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&sstr2) == 1);
#endif
  }

  assert(sstr1.str().size() == 1);
  assert(sstr1.str()[0] == CharT('A'));
  assert(sstr2.str() == expected);
}

template <class CharT>
static void test_short_write_after_swap() {
  std::basic_stringbuf<CharT> sstr1;
  std::basic_stringbuf<CharT> sstr2;
  std::basic_string<CharT> expected(42, CharT('*')); // a long string

  {
    std::basic_syncbuf<CharT> sync_buf1(&sstr1);
    sync_buf1.sputc(CharT('A')); // a short string

    std::basic_syncbuf<CharT> sync_buf2(&sstr2);
    sync_buf2.sputn(expected.data(), expected.size());

    sync_buf1.swap(sync_buf2);
    sync_buf1.sputc(CharT('B'));
    expected.push_back(CharT('B'));
    sync_buf2.sputc(CharT('Z'));

    assert(sstr1.str().empty());
    assert(sstr2.str().empty());
  }

  assert(sstr1.str().size() == 2);
  assert(sstr1.str()[0] == CharT('A'));
  assert(sstr1.str()[1] == CharT('Z'));
  assert(sstr2.str() == expected);
}

template <class CharT>
static void test_long_write_after_swap() {
  std::basic_stringbuf<CharT> sstr1;
  std::basic_stringbuf<CharT> sstr2;
  std::basic_string<CharT> expected(42, CharT('*')); // a long string

  {
    std::basic_syncbuf<CharT> sync_buf1(&sstr1);
    sync_buf1.sputc(CharT('A')); // a short string

    std::basic_syncbuf<CharT> sync_buf2(&sstr2);
    sync_buf2.sputn(expected.data(), expected.size());

    sync_buf1.swap(sync_buf2);
    sync_buf1.sputn(expected.data(), expected.size());
    sync_buf2.sputn(expected.data(), expected.size());

    assert(sstr1.str().empty());
    assert(sstr2.str().empty());
  }

  assert(sstr1.str().size() == 1 + expected.size());
  assert(sstr1.str()[0] == CharT('A'));
  assert(sstr1.str().substr(1) == expected);
  assert(sstr2.str() == expected + expected);
}

template <class CharT>
static void test_emit_on_sync() {
  { // false false

    std::basic_stringbuf<CharT> sstr1;
    std::basic_stringbuf<CharT> sstr2;
    std::basic_string<CharT> expected(42, CharT('*')); // a long string

    {
      std::basic_syncbuf<CharT> sync_buf1(&sstr1);
      sync_buf1.set_emit_on_sync(false);
      sync_buf1.sputc(CharT('A')); // a short string

      std::basic_syncbuf<CharT> sync_buf2(&sstr2);
      sync_buf2.set_emit_on_sync(false);
      sync_buf2.sputn(expected.data(), expected.size());

      sync_buf1.swap(sync_buf2);

      assert(sstr1.str().empty());
      assert(sstr2.str().empty());

      sync_buf1.pubsync();
      assert(sstr1.str().empty());
      assert(sstr2.str().empty());

      sync_buf2.pubsync();
      assert(sstr1.str().empty());
      assert(sstr2.str().empty());
    }

    assert(sstr1.str().size() == 1);
    assert(sstr1.str()[0] == CharT('A'));
    assert(sstr2.str() == expected);
  }

  { // false true

    std::basic_stringbuf<CharT> sstr1;
    std::basic_stringbuf<CharT> sstr2;
    std::basic_string<CharT> expected(42, CharT('*')); // a long string

    {
      std::basic_syncbuf<CharT> sync_buf1(&sstr1);
      sync_buf1.set_emit_on_sync(true);
      sync_buf1.sputc(CharT('A')); // a short string

      std::basic_syncbuf<CharT> sync_buf2(&sstr2);
      sync_buf2.set_emit_on_sync(false);
      sync_buf2.sputn(expected.data(), expected.size());

      sync_buf1.swap(sync_buf2);

      assert(sstr1.str().empty());
      assert(sstr2.str().empty());

      sync_buf1.pubsync();
      assert(sstr1.str().empty());
      assert(sstr2.str().empty());

      sync_buf2.pubsync();
      assert(sstr1.str().size() == 1);
      assert(sstr1.str()[0] == CharT('A'));
      assert(sstr2.str().empty());
    }

    assert(sstr1.str().size() == 1);
    assert(sstr1.str()[0] == CharT('A'));
    assert(sstr2.str() == expected);
  }

  { // true false

    std::basic_stringbuf<CharT> sstr1;
    std::basic_stringbuf<CharT> sstr2;
    std::basic_string<CharT> expected(42, CharT('*')); // a long string

    {
      std::basic_syncbuf<CharT> sync_buf1(&sstr1);
      sync_buf1.set_emit_on_sync(false);
      sync_buf1.sputc(CharT('A')); // a short string

      std::basic_syncbuf<CharT> sync_buf2(&sstr2);
      sync_buf2.set_emit_on_sync(true);
      sync_buf2.sputn(expected.data(), expected.size());

      sync_buf1.swap(sync_buf2);

      assert(sstr1.str().empty());
      assert(sstr2.str().empty());

      sync_buf1.pubsync();
      assert(sstr1.str().empty());
      assert(sstr2.str() == expected);

      sync_buf2.pubsync();
      assert(sstr1.str().empty());
      assert(sstr2.str() == expected);
    }

    assert(sstr1.str().size() == 1);
    assert(sstr1.str()[0] == CharT('A'));
    assert(sstr2.str() == expected);
  }

  { // true true

    std::basic_stringbuf<CharT> sstr1;
    std::basic_stringbuf<CharT> sstr2;
    std::basic_string<CharT> expected(42, CharT('*')); // a long string

    {
      std::basic_syncbuf<CharT> sync_buf1(&sstr1);
      sync_buf1.set_emit_on_sync(true);
      sync_buf1.sputc(CharT('A')); // a short string

      std::basic_syncbuf<CharT> sync_buf2(&sstr2);
      sync_buf2.set_emit_on_sync(true);
      sync_buf2.sputn(expected.data(), expected.size());

      sync_buf1.swap(sync_buf2);

      assert(sstr1.str().empty());
      assert(sstr2.str().empty());

      sync_buf1.pubsync();
      assert(sstr1.str().empty());
      assert(sstr2.str() == expected);

      sync_buf2.pubsync();
      assert(sstr1.str().size() == 1);
      assert(sstr1.str()[0] == CharT('A'));
      assert(sstr2.str() == expected);
    }

    assert(sstr1.str().size() == 1);
    assert(sstr1.str()[0] == CharT('A'));
    assert(sstr2.str() == expected);
  }
}

template <class CharT>
static void test() {
  test_basic<CharT>();
  test_emit_on_sync<CharT>();
  test_short_write_after_swap<CharT>();
  test_long_write_after_swap<CharT>();
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
