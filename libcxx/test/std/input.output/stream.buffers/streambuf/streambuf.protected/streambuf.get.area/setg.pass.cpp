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

// void setg(char_type* gbeg, char_type* gnext, char_type* gend);

#include <streambuf>
#include <cassert>

template <class CharT>
struct test
    : public std::basic_streambuf<CharT>
{
    typedef std::basic_streambuf<CharT> base;

    test() {}

    void setg(CharT* gbeg, CharT* gnext, CharT* gend)
    {
        base::setg(gbeg, gnext, gend);
        assert(base::eback() == gbeg);
        assert(base::gptr() == gnext);
        assert(base::egptr() == gend);
    }
};

int main(int, char**)
{
    {
        test<char> t;
        char in[] = "ABC";
        t.setg(in, in+1, in+sizeof(in)/sizeof(in[0]));
    }
    {
        test<wchar_t> t;
        wchar_t in[] = L"ABC";
        t.setg(in, in+1, in+sizeof(in)/sizeof(in[0]));
    }

  return 0;
}
