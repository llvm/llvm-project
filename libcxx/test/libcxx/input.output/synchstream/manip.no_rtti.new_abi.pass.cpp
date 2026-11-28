//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// REQUIRES: libcpp-abi-syncbuf-manip-virtual
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-experimental-syncstream

// ADDITIONAL_COMPILE_FLAGS: -fno-rtti

// With _LIBCPP_ABI_SYNCBUF_MANIP_VIRTUAL, the manipulators work without RTTI.

#include <cassert>
#include <memory>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <syncstream>

#include "test_macros.h"

template <class CharT>
struct user_buf : std::basic_streambuf<CharT> {
  int sync_count = 0;

protected:
  int sync() override {
    ++sync_count;
    return 0;
  }
};

template <class T>
struct user_alloc : std::allocator<T> {
  using value_type = T;
  user_alloc()     = default;
  template <class U>
  user_alloc(const user_alloc<U>&) {}
};

template <class CharT>
void test() {
  {
    // emit_on_flush
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss << std::emit_on_flush << 5;
    assert(buf.str().empty());
    ss.flush();
    assert(buf.str().size() == 1);
    assert(ss.good());
  }
  {
    // noemit_on_flush
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss.rdbuf()->set_emit_on_sync(true);
    ss << std::noemit_on_flush << 5;
    ss.flush();
    assert(buf.str().empty());
    ss.emit();
    assert(buf.str().size() == 1);
  }
  {
    // flush_emit emits, and does not change the emit-on-sync flag
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss << 5 << std::flush_emit;
    assert(buf.str().size() == 1);
    assert(ss.good());
    ss << 6;
    ss.flush();
    assert(buf.str().size() == 1);
  }
  {
    // flush_emit sets badbit if emit fails
    std::basic_osyncstream<CharT> ss(nullptr);
    ss << std::flush_emit;
    assert(ss.bad());
  }
  {
    // a plain basic_ostream whose buffer is a basic_syncbuf
    std::basic_stringbuf<CharT> buf;
    std::basic_syncbuf<CharT> sb(&buf);
    std::basic_ostream<CharT> os(&sb);
    os << std::emit_on_flush << 5;
    os.flush();
    assert(buf.str().size() == 1);
  }
  {
    // a basic_syncbuf with a different allocator
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT, std::char_traits<CharT>, user_alloc<CharT>> ss(&buf);
    ss << std::emit_on_flush << 5;
    ss.flush();
    assert(buf.str().size() == 1);
  }
  {
    // non sync streams are not affected
    std::basic_ostringstream<CharT> os;
    os << std::emit_on_flush << std::noemit_on_flush << 5 << std::flush_emit;
    assert(os.str().size() == 1);
    assert(os.good());
  }
  {
    // a stream buffer defined in -fno-rtti code
    user_buf<CharT> buf;
    std::basic_ostream<CharT> os(&buf);
    os << std::emit_on_flush << std::noemit_on_flush << std::flush_emit;
    assert(buf.sync_count == 1);
    assert(os.good());
  }
  {
    // a stream without a buffer
    std::basic_ostream<CharT> os(nullptr);
    os << std::emit_on_flush << std::noemit_on_flush << std::flush_emit;
    assert(os.bad());
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
