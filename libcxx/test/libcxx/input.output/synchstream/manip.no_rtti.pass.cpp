//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-experimental-syncstream

// ADDITIONAL_COMPILE_FLAGS: -fno-rtti

// emit_on_flush, noemit_on_flush and flush_emit need to detect whether a
// stream's buffer is a basic_syncbuf, which requires RTTI. Without RTTI,
// libc++ treats every stream as a non-sync stream: emit_on_flush and
// noemit_on_flush do nothing, and flush_emit only flushes.

#include <cassert>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <syncstream>

#include "test_macros.h"

// A stream buffer whose vtable is emitted in this -fno-rtti TU.
template <class CharT>
struct user_buf : std::basic_streambuf<CharT> {
  int sync_count = 0;

protected:
  int sync() override {
    ++sync_count;
    return 0;
  }
};

template <class CharT>
void test() {
  {
    // emit_on_flush has no effect
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss << std::emit_on_flush << 5;
    ss.flush();
    assert(buf.str().empty());
    ss.emit();
    assert(buf.str().size() == 1);
  }
  {
    // noemit_on_flush has no effect
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss.rdbuf()->set_emit_on_sync(true);
    ss << std::noemit_on_flush << 5;
    ss.flush();
    assert(buf.str().size() == 1);
  }
  {
    // flush_emit only flushes
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss << 5 << std::flush_emit;
    assert(buf.str().empty());
    assert(ss.good());
    ss.emit();
    assert(buf.str().size() == 1);
  }
  {
    // flush_emit does not emit, so a missing wrapped buffer does not set badbit
    std::basic_osyncstream<CharT> ss(nullptr);
    ss << std::flush_emit;
    assert(ss.good());
  }
  {
    // a stream buffer defined in -fno-rtti code
    user_buf<CharT> buf;
    std::basic_ostream<CharT> os(&buf);
    os << std::emit_on_flush << std::noemit_on_flush << std::flush_emit;
    assert(buf.sync_count == 1);
    assert(os.good());
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
