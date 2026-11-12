//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8

// <streambuf>

// template <class charT, class traits = char_traits<charT> >
// class basic_streambuf;

// basic_streambuf(const basic_streambuf& rhs);

#include <streambuf>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

template <class CharT>
struct test
    : public std::basic_streambuf<CharT>
{
    typedef std::basic_streambuf<CharT> base;
    test() {}

    test(const test& t)
        : std::basic_streambuf<CharT>(t)
    {
        assert(this->eback() == t.eback());
        assert(this->gptr()  == t.gptr());
        assert(this->egptr() == t.egptr());
        assert(this->pbase() == t.pbase());
        assert(this->pptr()  == t.pptr());
        assert(this->epptr() == t.epptr());
        assert(this->getloc() == t.getloc());
    }

    void setg(CharT* gbeg, CharT* gnext, CharT* gend)
    {
        base::setg(gbeg, gnext, gend);
    }
    void setp(CharT* pbeg, CharT* pend)
    {
        base::setp(pbeg, pend);
    }
};

int main(int, char**)
{
    {
        test<char> t;
        test<char> t2 = t;
    }
    {
        char g[3];
        char p[3];
        test<char> t;
        t.setg(&g[0], &g[1], &g[2]);
        t.setp(&p[0], &p[2]);
        test<char> t2 = t;
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        wchar_t g[3];
        wchar_t p[3];
        test<wchar_t> t;
        t.setg(&g[0], &g[1], &g[2]);
        t.setp(&p[0], &p[2]);
        test<wchar_t> t2 = t;
    }
    {
        test<wchar_t> t;
        test<wchar_t> t2 = t;
    }
#endif
    std::locale::global(std::locale(LOCALE_en_US_UTF_8));
    {
        test<char> t;
        test<char> t2 = t;
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        test<wchar_t> t;
        test<wchar_t> t2 = t;
    }
#endif

  return 0;
}
