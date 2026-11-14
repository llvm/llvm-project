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

// template<class charT, class traits>
// basic_ostream<charT, traits>& flush_emit(basic_ostream<charT, traits>& os);

#include <cassert>
#include <concepts>
#include <ios>
#include <ostream>
#include <sstream>
#include <syncstream>

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

#ifndef TEST_HAS_NO_EXCEPTIONS
template <class CharT>
struct throwing_buf : std::basic_stringbuf<CharT> {
  bool should_throw = true;

protected:
  std::streamsize xsputn(const CharT* s, std::streamsize n) override {
    if (should_throw)
      throw 42;
    return std::basic_stringbuf<CharT>::xsputn(s, n);
  }
};
#endif

template <class CharT>
void test() {
  {
    // non sync stream: just flush
    sync_counting_buf<CharT> buf;
    std::basic_ostream<CharT> os(&buf);
    os << 5;
    std::same_as<std::basic_ostream<CharT>&> decltype(auto) result = std::flush_emit(os);
    assert(&result == &os);
    assert(buf.sync_count == 1);
    assert(!buf.str().empty());
    assert(os.good());
  }
  {
    // non sync stream: << syntax
    sync_counting_buf<CharT> buf;
    std::basic_ostream<CharT> os(&buf);
    std::same_as<std::basic_ostream<CharT>&> decltype(auto) result = os << std::flush_emit;
    assert(&result == &os);
    assert(buf.sync_count == 1);
    assert(os.good());
  }
  {
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss << 5;
    assert(buf.str().empty());
    std::same_as<std::basic_ostream<CharT>&> decltype(auto) result = std::flush_emit(ss);
    assert(&result == &ss);
    assert(!buf.str().empty());
    assert(ss.good());
  }
  {
    // << syntax
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss << 5;
    assert(buf.str().empty());
    std::same_as<std::basic_ostream<CharT>&> decltype(auto) result = ss << std::flush_emit;
    assert(&result == &ss);
    assert(!buf.str().empty());
    assert(ss.good());
  }
  {
    // emit() fails because there is no wrapped stream buffer: badbit is set
    std::basic_osyncstream<CharT> ss(nullptr);
    assert(ss.good());
    std::same_as<std::basic_ostream<CharT>&> decltype(auto) result = std::flush_emit(ss);
    assert(&result == &ss);
    assert(ss.bad());
  }
  {
    // emit() fails, << syntax
    std::basic_osyncstream<CharT> ss(nullptr);
    std::same_as<std::basic_ostream<CharT>&> decltype(auto) result = ss << std::flush_emit;
    assert(&result == &ss);
    assert(ss.bad());
  }
  {
    // the sentry fails because the stream is not good: emit() is not called
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss << 5;
    ss.setstate(std::ios_base::failbit);
    std::flush_emit(ss);
    assert(buf.str().empty());
    assert(ss.fail());
    assert(!ss.bad());
    ss.clear();
    ss.emit();
    assert(!buf.str().empty());
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    // emit() fails and badbit is set in exceptions(): ios_base::failure is thrown
    std::basic_osyncstream<CharT> ss(nullptr);
    ss.exceptions(std::ios_base::badbit);
    try {
      std::flush_emit(ss);
      assert(false);
    } catch (const std::ios_base::failure&) {
    }
    assert(ss.bad());
  }
  {
    // emit() throws and badbit is not set in exceptions(): the exception is
    // swallowed, badbit is set, and emit-on-sync is restored
    throwing_buf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss << 5;
    std::flush_emit(ss);
    assert(ss.bad());

    buf.should_throw = false;
    ss.clear();
    ss.flush();
    assert(buf.str().empty());
    ss.emit();
    assert(!buf.str().empty());
  }
  {
    // emit() throws and badbit is set in exceptions(): the exception is
    // rethrown, and emit-on-sync is restored
    throwing_buf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss.exceptions(std::ios_base::badbit);
    ss << 5;
    try {
      std::flush_emit(ss);
      assert(false);
    } catch (int i) {
      assert(i == 42);
    }
    assert(ss.bad());

    buf.should_throw = false;
    ss.exceptions(std::ios_base::goodbit);
    ss.clear();
    ss.flush();
    assert(buf.str().empty());
    ss.emit();
    assert(!buf.str().empty());
  }
#endif
  {
    // flush_emit leaves emit-on-sync unchanged when it was false
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss << 5;
    std::flush_emit(ss);
    assert(buf.str().size() == 1);
    ss << 6;
    ss.flush();
    assert(buf.str().size() == 1);
  }
  {
    // flush_emit leaves emit-on-sync unchanged when it was true
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss << std::emit_on_flush << 5;
    std::flush_emit(ss);
    assert(buf.str().size() == 1);
    ss << 6;
    ss.flush();
    assert(buf.str().size() == 2);
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
