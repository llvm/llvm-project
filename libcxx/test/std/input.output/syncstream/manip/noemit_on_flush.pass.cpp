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
// basic_ostream<charT, traits>& noemit_on_flush(basic_ostream<charT, traits>& os);

#include <cassert>
#include <ostream>
#include <sstream>
#include <syncstream>

template <class CharT>
void test_noemit_on_flush() {
  {
    // non sync stream: nothing happens
    std::basic_ostringstream<CharT> os;
    std::noemit_on_flush(os);
  }
  {
    std::basic_stringbuf<CharT> buf;
    std::basic_osyncstream<CharT> ss(&buf);
    std::noemit_on_flush(ss);
    ss << 5;
    ss.flush();
    assert(buf.str().empty());
    ss.emit();
    assert(!buf.str().empty());
  }
}

int main(int, char**) {
  test_noemit_on_flush<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_noemit_on_flush<wchar_t>();
#endif

  return 0;
}
