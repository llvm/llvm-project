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
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
