//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <streambuf>

// template <class charT, class traits = char_traits<charT> >
// class basic_streambuf;

// void pbump(int n);
//
// REQUIRES: long_tests

#include <sstream>
#include <cassert>
#include "test_macros.h"

struct SB : std::stringbuf
{
  SB() : std::stringbuf(std::ios::ate|std::ios::out) { }
  const char* pubpbase() const { return pbase(); }
  const char* pubpptr() const { return pptr(); }
};

int main()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
#endif
        std::string str(2147483648, 'a');
        SB sb;
        sb.str(str);
        assert(sb.pubpbase() <= sb.pubpptr());
#ifndef TEST_HAS_NO_EXCEPTIONS
    }
    catch (const std::length_error &) {} // maybe the string can't take 2GB
    catch (const std::bad_alloc    &) {} // maybe we don't have enough RAM
#endif
}
