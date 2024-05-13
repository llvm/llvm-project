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

// Tests the inherited function using a custom allocator.
//
// streamsize basic_streambuf<charT, traits>::sputc(const char_type* s, streamsize n);
//
// This test also validates the observable behaviour after move assignment and
// construction. This test uses a large buffer so the underlying string is in
// long mode.

#include <array>
#include <syncstream>
#include <cassert>
#include <sstream>

#include "test_macros.h"
#include "test_allocator.h"

template <class CharT>
void test() {
  std::array< CharT, 17> input{
      CharT('a'),
      CharT('1'),
      CharT('+'),
      CharT('A'),
      CharT('g'),
      CharT('0'),
      CharT('@'),
      CharT('Z'),
      CharT('q'),
      CharT('8'),
      CharT('#'),
      CharT('D'),
      CharT('t'),
      CharT('9'),
      CharT('$'),
      CharT('A'),
      CharT(' ')};
  std::basic_string<CharT> expected;
  for (int i = 0; i < 1024; ++i)
    expected.push_back(input[i % input.size()]);

  using SyncBuf = std::basic_syncbuf<CharT, std::char_traits<CharT>, test_allocator<CharT>>;
  { // Normal
    std::basic_stringbuf<CharT> buf;
    test_allocator_statistics stats;
    test_allocator<CharT> allocator{&stats};

    {
      SyncBuf sync_buf{&buf, allocator};
      std::streamsize ret = sync_buf.sputn(expected.data(), expected.size());
      assert(ret == 1024);

      // The synchronization happens upon destruction of sync_buf.
      assert(buf.str().empty());
      assert(stats.allocated_size >= 1024);
    }
    assert(buf.str() == expected);
    assert(stats.allocated_size == 0);
  }
  { // Move construction
    std::basic_stringbuf<CharT> buf;
    test_allocator_statistics stats;
    test_allocator<CharT> allocator{&stats};

    {
      SyncBuf sync_buf{&buf, allocator};
      std::streamsize ret = sync_buf.sputn(expected.data(), expected.size());
      assert(ret == 1024);
      {
        SyncBuf new_sync_buf{std::move(sync_buf)};
        ret = new_sync_buf.sputn(expected.data(), expected.size());
        assert(ret == 1024);

        // The synchronization happens upon destruction of new_sync_buf.
        assert(buf.str().empty());
        assert(stats.allocated_size >= 2048);
      }
      assert(buf.str() == expected + expected);
      assert(stats.allocated_size == 0);
    }
    assert(buf.str() == expected + expected);
    assert(stats.allocated_size == 0);
  }
  { // Move assignment non-propagating allocator
    std::basic_stringbuf<CharT> buf;
    test_allocator_statistics stats;
    test_allocator<CharT> allocator{&stats};
    static_assert(!std::allocator_traits<test_allocator<CharT>>::propagate_on_container_move_assignment::value);

    {
      SyncBuf sync_buf{&buf, allocator};
      std::streamsize ret = sync_buf.sputn(expected.data(), expected.size());
      assert(ret == 1024);
      {
        SyncBuf new_sync_buf;
        test_allocator<CharT> a = new_sync_buf.get_allocator();
        new_sync_buf            = std::move(sync_buf);
        assert(new_sync_buf.get_allocator() == a);

        ret = new_sync_buf.sputn(expected.data(), expected.size());
        assert(ret == 1024);

        // The synchronization happens upon destruction of new_sync_buf.
        assert(buf.str().empty());
      }
      assert(buf.str() == expected + expected);
    }
    assert(buf.str() == expected + expected);
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  return 0;
}
