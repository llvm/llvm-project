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

// streamsize sgetn(char_type* s, streamsize n);

#include <streambuf>
#include <cassert>

int xsgetn_called = 0;

struct test
    : public std::basic_streambuf<char>
{
    test() {}

protected:
    std::streamsize xsgetn(char_type*, std::streamsize)
    {
        ++xsgetn_called;
        return 10;
    }
};

int main(int, char**)
{
    test t;
    assert(xsgetn_called == 0);
    assert(t.sgetn(0, 0) == 10);
    assert(xsgetn_called == 1);

  return 0;
}
