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

// int_type sbumpc();

#include <streambuf>
#include <cassert>

int uflow_called = 0;

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
    int_type uflow()
    {
        ++uflow_called;
        return 'a';
    }
};

int main()
{
    {
        test t;
        assert(uflow_called == 0);
        assert(t.sbumpc() == 'a');
        assert(uflow_called == 1);
        char in[] = "ABC";
        t.setg(in, in, in+sizeof(in));
        assert(t.sbumpc() == 'A');
        assert(uflow_called == 1);
        assert(t.sbumpc() == 'B');
        assert(uflow_called == 1);
    }
}
