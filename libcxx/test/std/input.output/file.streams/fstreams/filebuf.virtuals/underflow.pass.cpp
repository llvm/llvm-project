//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8
// FILE_DEPENDENCIES: underflow.dat, underflow_utf8.dat

// <fstream>

// int_type underflow();

// This test is not entirely portable

#include <fstream>
#include <cstddef>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

template <class CharT>
struct test_buf
    : public std::basic_filebuf<CharT>
{
    typedef std::basic_filebuf<CharT> base;
    typedef typename base::char_type  char_type;
    typedef typename base::int_type   int_type;

    char_type* eback() const {return base::eback();}
    char_type* gptr()  const {return base::gptr();}
    char_type* egptr() const {return base::egptr();}
    void gbump(int n) {base::gbump(n);}

    virtual int_type underflow() {return base::underflow();}
};

int main(int, char**)
{
    {
        test_buf<char> f;
        assert(f.open("underflow.dat", std::ios_base::in) != 0);
        assert(f.is_open());
        assert(f.eback() == 0);
        assert(f.gptr() == 0);
        assert(f.egptr() == 0);
        assert(f.underflow() == '1');
        assert(f.eback() != 0);
        assert(f.eback() == f.gptr());
        assert(*f.gptr() == '1');
        assert(f.egptr() - f.eback() == 9);
    }
    {
        test_buf<char> f;
        assert(f.open("underflow.dat", std::ios_base::in) != 0);
        assert(f.pubsetbuf(0, 0));
        assert(f.is_open());
        assert(f.eback() == 0);
        assert(f.gptr() == 0);
        assert(f.egptr() == 0);
        assert(f.underflow() == '1');
        assert(f.eback() != 0);
        assert(f.eback() == f.gptr());
        assert(*f.gptr() == '1');
        assert(f.egptr() - f.eback() == 8);
        f.gbump(8);
        assert(f.sgetc() == '9');
        assert(f.eback()[0] == '5');
        assert(f.eback()[1] == '6');
        assert(f.eback()[2] == '7');
        assert(f.eback()[3] == '8');
        assert(f.gptr() - f.eback() == 4);
        assert(*f.gptr() == '9');
        assert(f.egptr() - f.gptr() == 1);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        test_buf<wchar_t> f;
        assert(f.open("underflow.dat", std::ios_base::in) != 0);
        assert(f.is_open());
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
        test_buf<wchar_t> f;
        assert(f.pubsetbuf(0, 0));
        assert(f.open("underflow.dat", std::ios_base::in) != 0);
        assert(f.is_open());
        assert(f.eback() == 0);
        assert(f.gptr() == 0);
        assert(f.egptr() == 0);
        assert(f.underflow() == L'1');
        assert(f.eback() != 0);
        assert(f.eback() == f.gptr());
        assert(*f.gptr() == L'1');
        assert(f.egptr() - f.eback() == 8);
        f.gbump(8);
        assert(f.sgetc() == L'9');
        assert(f.eback()[0] == L'5');
        assert(f.eback()[1] == L'6');
        assert(f.eback()[2] == L'7');
        assert(f.eback()[3] == L'8');
        assert(f.gptr() - f.eback() == 4);
        assert(*f.gptr() == L'9');
        assert(f.egptr() - f.gptr() == 1);
    }
    {
        typedef std::char_traits<wchar_t> Traits;
        test_buf<wchar_t> f;
        f.pubimbue(std::locale(LOCALE_en_US_UTF_8));
        assert(f.open("underflow_utf8.dat", std::ios_base::in) != 0);
        assert(f.is_open());
        assert(f.sbumpc() == 0x4E51);
        assert(f.sbumpc() == 0x4E52);
        assert(f.sbumpc() == 0x4E53);
        assert(f.sbumpc() == static_cast<Traits::int_type>(-1));
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
