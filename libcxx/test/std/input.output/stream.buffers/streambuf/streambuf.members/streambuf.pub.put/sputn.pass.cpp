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

// streamsize sputn(const char_type* s, streamsize n);

#include <streambuf>
#include <cassert>

int xsputn_called = 0;

struct test
    : public std::basic_streambuf<char>
{
    test() {}

protected:
    std::streamsize xsputn(const char_type*, std::streamsize)
    {
        ++xsputn_called;
        return 5;
    }
};

int main()
{
    test t;
    assert(xsputn_called == 0);
    assert(t.sputn(0, 0) == 5);
    assert(xsputn_called == 1);
}
