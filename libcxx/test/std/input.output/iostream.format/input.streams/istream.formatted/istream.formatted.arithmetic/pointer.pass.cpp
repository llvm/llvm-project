//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: with_system_cxx_lib=macosx10.14
// XFAIL: with_system_cxx_lib=macosx10.13
// XFAIL: with_system_cxx_lib=macosx10.12
// XFAIL: with_system_cxx_lib=macosx10.11
// XFAIL: with_system_cxx_lib=macosx10.10
// XFAIL: with_system_cxx_lib=macosx10.9

// <istream>

// template <class charT, class traits = char_traits<charT> >
//   class basic_istream;

// operator>>(void*& val);

#include <istream>
#include <cassert>
#include "test_macros.h"

template <class CharT>
struct testbuf
    : public std::basic_streambuf<CharT>
{
    typedef std::basic_string<CharT> string_type;
    typedef std::basic_streambuf<CharT> base;
private:
    string_type str_;
public:

    testbuf() {}
    testbuf(const string_type& str)
        : str_(str)
    {
        base::setg(const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()) + str_.size());
    }

    CharT* eback() const {return base::eback();}
    CharT* gptr() const {return base::gptr();}
    CharT* egptr() const {return base::egptr();}
};

int main(int, char**)
{
    {
        std::istream is((std::streambuf*)0);
        void* n = 0;
        is >> n;
        assert(is.fail());
    }
    {
        testbuf<char> sb("0");
        std::istream is(&sb);
        void* n = (void*)1;
        is >> n;
        assert(n == 0);
        assert( is.eof());
        assert(!is.fail());
    }
    {
        testbuf<char> sb(" 1 ");
        std::istream is(&sb);
        void* n = 0;
        is >> n;
        assert(n == (void*)1);
        assert(!is.eof());
        assert(!is.fail());
    }
    {
        testbuf<wchar_t> sb(L" 1 ");
        std::wistream is(&sb);
        void* n = 0;
        is >> n;
        assert(n == (void*)1);
        assert(!is.eof());
        assert(!is.fail());
    }
    {
        testbuf<char> sb("12345678");
        std::istream is(&sb);
        void* n = 0;
        is >> n;
        assert(n == (void*)0x12345678);
        assert( is.eof());
        assert(!is.fail());
    }
    {
        testbuf<wchar_t> sb(L"12345678");
        std::wistream is(&sb);
        void* n = 0;
        is >> n;
        assert(n == (void*)0x12345678);
        assert( is.eof());
        assert(!is.fail());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        testbuf<char> sb;
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios_base::failbit);

        bool threw = false;
        try {
            void* n = 0;
            is >> n;
        } catch (std::ios_base::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(is.fail());
        assert(is.eof());
        assert(threw);
    }
    {
        testbuf<char> sb;
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios_base::eofbit);

        bool threw = false;
        try {
            void* n = 0;
            is >> n;
        } catch (std::ios_base::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(is.fail());
        assert(is.eof());
        assert(threw);
    }
#endif // TEST_HAS_NO_EXCEPTIONS

    return 0;
}
