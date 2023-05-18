//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX26_REMOVED_CODECVT -D_LIBCPP_ENABLE_CXX26_REMOVED_WSTRING_CONVERT

// wbuffer_convert<Codecvt, Elem, Tr>

// int_type underflow();

// This test is not entirely portable

// XFAIL: no-wide-characters

#include <locale>
#include <cassert>
#include <codecvt>
#include <sstream>

#include "test_macros.h"

struct test_buf
    : public std::wbuffer_convert<std::codecvt_utf8<wchar_t> >
{
    typedef std::wbuffer_convert<std::codecvt_utf8<wchar_t> > base;
    typedef base::char_type   char_type;
    typedef base::int_type    int_type;
    typedef base::traits_type traits_type;

    explicit test_buf(std::streambuf* sb) : base(sb) {}

    char_type* eback() const {return base::eback();}
    char_type* gptr()  const {return base::gptr();}
    char_type* egptr() const {return base::egptr();}
    void gbump(int n) {base::gbump(n);}

    virtual int_type underflow() {return base::underflow();}
};

int main(int, char**)
{
    {
        std::string s = "123456789";
        std::istringstream bs(s);
        test_buf f(bs.rdbuf());
        assert(f.eback() == 0);
        assert(f.gptr() == 0);
        assert(f.egptr() == 0);
        assert(f.underflow() == L'1');
        assert(f.eback() != 0);
        assert(f.eback() == f.gptr());
        assert(*f.gptr() == L'1');
        assert(f.egptr() - f.eback() == 9);
    }
    {
        std::string s = "123456789";
        std::istringstream bs(s);
        test_buf f(bs.rdbuf());
        assert(f.eback() == 0);
        assert(f.gptr() == 0);
        assert(f.egptr() == 0);
        assert(f.underflow() == L'1');
        assert(f.eback() != 0);
        assert(f.eback() == f.gptr());
        assert(*f.gptr() == L'1');
        assert(f.egptr() - f.eback() == 9);
        f.gbump(8);
        assert(f.sgetc() == L'9');
        assert(f.eback()[0] == L'1');
        assert(f.eback()[1] == L'2');
        assert(f.eback()[2] == L'3');
        assert(f.eback()[3] == L'4');
        assert(f.gptr() - f.eback() == 8);
        assert(*f.gptr() == L'9');
        assert(f.egptr() - f.gptr() == 1);
    }
    {
        std::string s = "乑乒乓";
        std::istringstream bs(s);
        test_buf f(bs.rdbuf());
        assert(f.sbumpc() == 0x4E51);
        assert(f.sbumpc() == 0x4E52);
        assert(f.sbumpc() == 0x4E53);
        assert(f.sbumpc() == test_buf::traits_type::eof());
    }

    return 0;
}
