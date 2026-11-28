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

// int sync() override;
//   Effects: Records that the wrapped stream buffer is to be flushed. Then, if
//   emit_on_sync is true, calls emit().
//
// bool emit();
//   ... If a call to sync since the most recent call to emit happened, then
//   calls wrapped->pubsync().

#include <cassert>
#include <ostream>
#include <sstream>
#include <syncstream>
#include <utility>

#include "test_macros.h"

template <class CharT>
struct sync_counting_buf : std::basic_stringbuf<CharT> {
  int sync_count = 0;

protected:
  int sync() override {
    ++sync_count;
    return std::basic_stringbuf<CharT>::sync();
  }
};

template <class CharT>
void test() {
  {
    // emit_on_sync == false: sync() only records the flush, emit() performs it
    sync_counting_buf<CharT> wrapped;
    std::basic_syncbuf<CharT> buf(&wrapped);
    std::basic_ostream<CharT> out(&buf);

    out << CharT('a');
    out.flush();
    assert(wrapped.str().empty());
    assert(wrapped.sync_count == 0);

    assert(buf.emit());
    assert(wrapped.str().size() == 1);
    assert(wrapped.sync_count == 1);

    // No sync() since the last emit(): emit() must not flush again.
    out << CharT('b');
    assert(buf.emit());
    assert(wrapped.str().size() == 2);
    assert(wrapped.sync_count == 1);
  }
  {
    // emit() without any preceding sync() must not flush the wrapped buffer
    sync_counting_buf<CharT> wrapped;
    std::basic_syncbuf<CharT> buf(&wrapped);
    std::basic_ostream<CharT> out(&buf);
    out << CharT('a');
    assert(buf.emit());
    assert(wrapped.sync_count == 0);
  }
  {
    // emit_on_sync == true: sync() calls emit(), which flushes immediately
    sync_counting_buf<CharT> wrapped;
    std::basic_syncbuf<CharT> buf(&wrapped);
    buf.set_emit_on_sync(true);
    std::basic_ostream<CharT> out(&buf);
    out << CharT('a');
    out.flush();
    assert(wrapped.str().size() == 1);
    assert(wrapped.sync_count == 1);
  }
  {
    // the destructor calls emit(), which must honour a pending flush
    sync_counting_buf<CharT> wrapped;
    {
      std::basic_syncbuf<CharT> buf(&wrapped);
      std::basic_ostream<CharT> out(&buf);
      out << CharT('a');
      out.flush();
      assert(wrapped.sync_count == 0);
    }
    assert(wrapped.str().size() == 1);
    assert(wrapped.sync_count == 1);
  }
  {
    // a pending flush is transferred by move construction
    sync_counting_buf<CharT> wrapped;
    std::basic_syncbuf<CharT> buf(&wrapped);
    std::basic_ostream<CharT> out(&buf);
    out << CharT('a');
    out.flush();
    std::basic_syncbuf<CharT> moved(std::move(buf));
    assert(moved.emit());
    assert(wrapped.sync_count == 1);
  }
  {
    // a pending flush is transferred by move assignment
    sync_counting_buf<CharT> wrapped;
    std::basic_syncbuf<CharT> buf(&wrapped);
    std::basic_ostream<CharT> out(&buf);
    out << CharT('a');
    out.flush();
    sync_counting_buf<CharT> other_wrapped;
    std::basic_syncbuf<CharT> moved(&other_wrapped);
    moved = std::move(buf);
    assert(moved.emit());
    assert(wrapped.sync_count == 1);
    assert(other_wrapped.sync_count == 0);
  }
  {
    // a pending flush is exchanged by swap
    sync_counting_buf<CharT> wrapped1;
    sync_counting_buf<CharT> wrapped2;
    std::basic_syncbuf<CharT> buf1(&wrapped1);
    std::basic_syncbuf<CharT> buf2(&wrapped2);
    std::basic_ostream<CharT> out(&buf1);
    out << CharT('a');
    out.flush();
    buf1.swap(buf2);
    assert(buf1.emit()); // now wraps wrapped2, nothing pending
    assert(wrapped2.sync_count == 0);
    assert(buf2.emit()); // now wraps wrapped1, flush pending
    assert(wrapped1.sync_count == 1);
  }
  {
    // osyncstream: flush() then emit()
    sync_counting_buf<CharT> wrapped;
    std::basic_osyncstream<CharT> ss(&wrapped);
    ss << CharT('a');
    ss.flush();
    ss.emit();
    assert(wrapped.str().size() == 1);
    assert(wrapped.sync_count == 1);
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
