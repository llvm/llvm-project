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

// int_type snextc();

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

int main(int, char**)
{
    {
        test t;
        assert(uflow_called == 0);
        assert(t.snextc() == -1);
        assert(uflow_called == 1);
        char in[] = "ABC";
        t.setg(in, in, in+sizeof(in));
        assert(t.snextc() == 'B');
        assert(uflow_called == 1);
        assert(t.snextc() == 'C');
        assert(uflow_called == 1);
    }

  return 0;
}
