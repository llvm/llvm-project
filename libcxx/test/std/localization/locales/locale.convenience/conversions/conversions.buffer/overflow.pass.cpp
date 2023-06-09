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

// int_type overflow(int_type c = traits::eof());

// This test is not entirely portable

// XFAIL: no-wide-characters

#include <locale>
#include <cassert>
#include <codecvt>
#include <fstream>
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

    char_type* pbase() const {return base::pbase();}
    char_type* pptr()  const {return base::pptr();}
    char_type* epptr() const {return base::epptr();}
    void gbump(int n) {base::gbump(n);}

    virtual int_type overflow(int_type c = traits_type::eof()) {return base::overflow(c);}
};

int main(int, char**)
{
    {
        std::string s;
        {
            std::ostringstream out;
            test_buf f(out.rdbuf());
            assert(f.pbase() == 0);
            assert(f.pptr() == 0);
            assert(f.epptr() == 0);
            assert(f.overflow(L'a') == L'a');
            assert(f.pbase() != 0);
            assert(f.pptr() == f.pbase());
            assert(f.epptr() - f.pbase() == 4095);
            s = out.str();
        }
        {
            std::istringstream in(s);
            test_buf f(in.rdbuf());
            assert(f.sgetc() == L'a');
        }
    }
    {
        std::string s;
        {
            std::ostringstream out;
            test_buf f(out.rdbuf());
            f.pubsetbuf(0, 0);
            assert(f.pbase() == 0);
            assert(f.pptr() == 0);
            assert(f.epptr() == 0);
            assert(f.overflow('a') == 'a');
            assert(f.pbase() == 0);
            assert(f.pptr() == 0);
            assert(f.epptr() == 0);
            s = out.str();
        }
        {
            std::istringstream in(s);
            test_buf f(in.rdbuf());
            assert(f.sgetc() == L'a');
        }
    }
    // TODO: Move this to std::stringstream once https://llvm.org/PR59083 has been resolved
#ifndef TEST_HAS_NO_FILESYSTEM
    {
        {
            std::ofstream bs("overflow.dat");
            test_buf f(bs.rdbuf());
            assert(f.sputc(0x4E51) == 0x4E51);
            assert(f.sputc(0x4E52) == 0x4E52);
            assert(f.sputc(0x4E53) == 0x4E53);
        }
        {
            std::ifstream f("overflow.dat");
            assert(f.is_open());
            assert(f.get() == 0xE4);
            assert(f.get() == 0xB9);
            assert(f.get() == 0x91);
            assert(f.get() == 0xE4);
            assert(f.get() == 0xB9);
            assert(f.get() == 0x92);
            assert(f.get() == 0xE4);
            assert(f.get() == 0xB9);
            assert(f.get() == 0x93);
            assert(f.get() == -1);
        }
        std::remove("overflow.dat");
    }
#endif // TEST_HAS_NO_FILESYSTEM

    return 0;
}
