//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <istream>

// template <class charT, class traits = char_traits<charT> >
// class basic_istream;

// basic_istream(basic_istream&& rhs);

#include <istream>
#include <cassert>
#include <streambuf>

#include "test_macros.h"

template <class CharT>
struct testbuf
    : public std::basic_streambuf<CharT>
{
    testbuf() {}
};

template <class CharT>
struct test_istream
    : public std::basic_istream<CharT>
{
    typedef std::basic_istream<CharT> base;
    test_istream(testbuf<CharT>* sb) : base(sb) {}

    test_istream(test_istream&& s)
        : base(std::move(s)) {}
};

int main(int, char**)
{
    {
        testbuf<char> sb;
        test_istream<char> is1(&sb);
        test_istream<char> is(std::move(is1));
        assert(is1.rdbuf() == &sb);
        assert(is1.gcount() == 0);
        assert(is.gcount() == 0);
        assert(is.rdbuf() == 0);
        assert(is.tie() == 0);
        assert(is.fill() == ' ');
        assert(is.rdstate() == is.goodbit);
        assert(is.exceptions() == is.goodbit);
        assert(is.flags() == (is.skipws | is.dec));
        assert(is.precision() == 6);
        assert(is.getloc().name() == "C");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb;
        test_istream<wchar_t> is1(&sb);
        test_istream<wchar_t> is(std::move(is1));
        assert(is1.gcount() == 0);
        assert(is.gcount() == 0);
        assert(is1.rdbuf() == &sb);
        assert(is.rdbuf() == 0);
        assert(is.tie() == 0);
        assert(is.fill() == L' ');
        assert(is.rdstate() == is.goodbit);
        assert(is.exceptions() == is.goodbit);
        assert(is.flags() == (is.skipws | is.dec));
        assert(is.precision() == 6);
        assert(is.getloc().name() == "C");
    }
#endif

  return 0;
}
