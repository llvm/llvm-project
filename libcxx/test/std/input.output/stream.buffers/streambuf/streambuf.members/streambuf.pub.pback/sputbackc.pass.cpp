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

// int_type sputbackc(char_type c);

#include <streambuf>
#include <cassert>

int pbackfail_called = 0;

struct test
    : public std::basic_streambuf<char>
{
    typedef std::basic_streambuf<char> base;

    test() {}

    void setg(char* gbeg, char* gnext, char* gend)
    {
        base::setg(gbeg, gnext, gend);
    }

protected:
    int_type pbackfail(int_type = traits_type::eof())
    {
        ++pbackfail_called;
        return 'a';
    }
};

int main()
{
    {
        test t;
        assert(pbackfail_called == 0);
        assert(t.sputbackc('A') == 'a');
        assert(pbackfail_called == 1);
        char in[] = "ABC";
        t.setg(in, in+1, in+sizeof(in));
        assert(t.sputbackc('A') == 'A');
        assert(pbackfail_called == 1);
        assert(t.sputbackc('A') == 'a');
        assert(pbackfail_called == 2);
    }
}
