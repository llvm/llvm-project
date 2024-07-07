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
#include <ostream>
#include <sstream>
#include <syncstream>

template <class CharT>
void test_flush_emit() {
  {
    // non sync stream: just flush
    std::basic_ostringstream<CharT> os;
    os << 5;
    std::flush_emit(os);
    assert(!os.rdbuf()->str().empty());
  }
  {
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    ss << 5;
    assert(buf.str().empty());
    std::flush_emit(ss);
    assert(!buf.str().empty());
  }
}

int main(int, char**) {
  test_flush_emit<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_flush_emit<wchar_t>();
#endif

  return 0;
}
