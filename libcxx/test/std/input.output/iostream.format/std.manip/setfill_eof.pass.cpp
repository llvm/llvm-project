//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that weof as a wchar_t value can be set as the fill character.

// UNSUPPORTED: no-wide-characters
// REQUIRES: target=powerpc{{(64)?}}-ibm-aix || target=s390x-ibm-zos

#include <iomanip>
#include <ostream>
#include <cassert>
#include <string>

template <class CharT>
struct testbuf : public std::basic_streambuf<CharT> {
  testbuf() {}
};

int main(int, char**) {
  testbuf<wchar_t> sb;
  std::wostream os(&sb);
  os << std::setfill((wchar_t)std::char_traits<wchar_t>::eof());
  assert(os.fill() == (wchar_t)std::char_traits<wchar_t>::eof());

  return 0;
}
