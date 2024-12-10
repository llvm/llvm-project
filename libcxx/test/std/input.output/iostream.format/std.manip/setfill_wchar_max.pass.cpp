//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that WCHAR_MAX as a wchar_t value can be set as the fill character.

// UNSUPPORTED: no-wide-characters

// Expect the test case to fail on targets where WEOF is the same as
// WCHAR_MAX with the libcpp ABI version 1 implementation. The libcpp ABI
// version 2 implementation fixes the problem.

// XFAIL: target={{.*}}-windows{{.*}} && libcpp-abi-version=1
// XFAIL: target=armv{{7|8}}{{l?}}{{.*}}-linux-gnueabihf && libcpp-abi-version=1
// XFAIL: target=aarch64{{.*}}-linux-gnu && libcpp-abi-version=1

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
