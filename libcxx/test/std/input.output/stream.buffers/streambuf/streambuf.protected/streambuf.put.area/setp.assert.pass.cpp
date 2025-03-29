//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <streambuf>

// template <class charT, class traits = char_traits<charT> >
// class basic_streambuf;

// void setp(char_type* pbeg, char_type* pend);

#include <algorithm>
#include <iterator>
#include <streambuf>
#include <string>

#include "check_assertion.h"
#include "make_string.h"
#include "test_macros.h"

template <class CharT>
struct streambuf : public std::basic_streambuf<CharT> {
  typedef std::basic_streambuf<CharT> base;

  streambuf() {}

  void setp(CharT* pbeg, CharT* pend) { base::setp(pbeg, pend); }
};

template <class CharT>
void test() {
  std::basic_string<CharT> str = MAKE_STRING(CharT, "ABCDEF");
  CharT arr[6];
  std::copy(str.begin(), str.end(), arr);

  {
    streambuf<CharT> buff;
    TEST_LIBCPP_ASSERT_FAILURE(buff.setp(std::begin(arr) + 3, std::begin(arr)), "[pbeg, pend) must be a valid range");
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
