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
// basic_ostream<charT, traits>& emit_on_flush(basic_ostream<charT, traits>& os);

#include <cassert>
#include <concepts>
#include <ostream>
#include <sstream>
#include <syncstream>

#include "test_macros.h"

template <class CharT>
void test() {
  {
    // non sync stream: nothing happens
    std::basic_ostringstream<CharT> os;
    std::same_as<std::basic_ostream<CharT>&> decltype(auto) result = std::emit_on_flush(os);
    assert(&result == &os);
    assert(os.good());
  }
  {
    // non sync stream: << syntax
    std::basic_ostringstream<CharT> os;
    std::same_as<std::basic_ostream<CharT>&> decltype(auto) result = os << std::emit_on_flush;
    assert(&result == &os);
    assert(os.good());
  }
  {
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    std::same_as<std::basic_ostream<CharT>&> decltype(auto) result = std::emit_on_flush(ss);
    assert(&result == &ss);
    ss << 5;
    assert(buf.str().empty());
    ss.flush();
    assert(!buf.str().empty());
    assert(ss.good());
  }
  {
    // << syntax
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    std::same_as<std::basic_ostream<CharT>&> decltype(auto) result = ss << std::emit_on_flush;
    assert(&result == &ss);
    ss << 5;
    assert(buf.str().empty());
    ss << std::flush;
    assert(!buf.str().empty());
    assert(ss.good());
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
