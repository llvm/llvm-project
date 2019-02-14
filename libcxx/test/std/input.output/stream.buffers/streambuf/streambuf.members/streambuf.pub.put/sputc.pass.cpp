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

// int_type sputc(char_type c);

#include <streambuf>
#include <cassert>

int overflow_called = 0;

struct test
    : public std::basic_streambuf<char>
{
    typedef std::basic_streambuf<char> base;

    test() {}

    void setg(char* gbeg, char* gnext, char* gend)
    {
        base::setg(gbeg, gnext, gend);
    }
    void setp(char* pbeg, char* pend)
    {
        base::setp(pbeg, pend);
    }

protected:
    int_type overflow(int_type = traits_type::eof())
    {
        ++overflow_called;
        return 'a';
    }
};

int main(int, char**)
{
    {
        test t;
        assert(overflow_called == 0);
        assert(t.sputc('A') == 'a');
        assert(overflow_called == 1);
        char out[3] = {0};
        t.setp(out, out+sizeof(out));
        assert(t.sputc('A') == 'A');
        assert(overflow_called == 1);
        assert(out[0] == 'A');
        assert(t.sputc('B') == 'B');
        assert(overflow_called == 1);
        assert(out[0] == 'A');
        assert(out[1] == 'B');
    }

  return 0;
}
