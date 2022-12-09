//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// wbuffer_convert<Codecvt, Elem, Tr>

// int_type pbackfail(int_type c = traits::eof());

// This test is not entirely portable

// XFAIL: no-wide-characters

#include <locale>
#include <cassert>
#include <codecvt>
#include <sstream>

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

    virtual int_type pbackfail(int_type c = traits_type::eof()) {return base::pbackfail(c);}
};

int main(int, char**)
{
    std::string const s = "123456789";
    {
        std::istringstream in(s);
        test_buf f(in.rdbuf());
        assert(f.sbumpc() == L'1');
        assert(f.sgetc() == L'2');
        assert(f.pbackfail(L'a') == test_buf::traits_type::eof());
    }
    {
        std::istringstream in(s);
        test_buf f(in.rdbuf());
        assert(f.sbumpc() == L'1');
        assert(f.sgetc() == L'2');
        assert(f.pbackfail(L'a') == test_buf::traits_type::eof());
        assert(f.sbumpc() == L'2');
        assert(f.sgetc() == L'3');
    }

  return 0;
}
