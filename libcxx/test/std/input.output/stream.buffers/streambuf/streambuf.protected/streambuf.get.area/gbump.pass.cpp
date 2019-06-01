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

// void gbump(int n);

#include <streambuf>
#include <cassert>

#include "test_macros.h"

template <class CharT>
struct test
    : public std::basic_streambuf<CharT>
{
    typedef std::basic_streambuf<CharT> base;

    test() {}

    void setg(CharT* gbeg, CharT* gnext, CharT* gend)
    {
        base::setg(gbeg, gnext, gend);
    }

    void gbump(int n)
    {
        CharT* gbeg = base::eback();
        CharT* gnext = base::gptr();
        CharT* gend = base::egptr();
        base::gbump(n);
        assert(base::eback() == gbeg);
        assert(base::gptr() == gnext+n);
        assert(base::egptr() == gend);
    }
};

int main(int, char**)
{
    {
        test<char> t;
        char in[] = "ABCDE";
        t.setg(in, in+1, in+sizeof(in)/sizeof(in[0]));
        t.gbump(2);
    }
    {
        test<wchar_t> t;
        wchar_t in[] = L"ABCDE";
        t.setg(in, in+1, in+sizeof(in)/sizeof(in[0]));
        t.gbump(3);
    }

  return 0;
}
