//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that WCHAR_MAX as a wchar_t value can be set as the fill character.

// UNSUPPORTED: no-wide-characters

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
  os << std::setfill((wchar_t)WCHAR_MAX);
  assert(os.fill() == (wchar_t)WCHAR_MAX);

  return 0;
}
