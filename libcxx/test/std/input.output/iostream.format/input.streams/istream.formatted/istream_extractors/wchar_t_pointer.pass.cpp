//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <istream>

// template<class charT, class traits>
//   basic_istream<charT,traits>& operator>>(basic_istream<charT,traits>&& in, charT* s);

#include <istream>
#include <cassert>
#include <streambuf>

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
        testbuf<char> sb("   abcdefghijk    ");
        std::istream is(&sb);
        char s[20];
        is >> s;
        assert(!is.eof());
        assert(!is.fail());
        assert(std::string(s) == "abcdefghijk");
    }
#if TEST_STD_VER > 17
    {
        testbuf<char> sb("   abcdefghijk    ");
        std::istream is(&sb);
        char s[4];
        is >> s;
        assert(!is.eof());
        assert(!is.fail());
        assert(std::string(s) == "abc");
    }
#endif
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L"   abcdefghijk    ");
        std::wistream is(&sb);
        is.width(4);
        wchar_t s[20];
        is >> s;
        assert(!is.eof());
        assert(!is.fail());
        assert(std::wstring(s) == L"abc");
        assert(is.width() == 0);
    }
    {
        testbuf<wchar_t> sb(L"   abcdefghijk");
        std::wistream is(&sb);
        wchar_t s[20];
        is >> s;
        assert( is.eof());
        assert(!is.fail());
        assert(std::wstring(s) == L"abcdefghijk");
        assert(is.width() == 0);
    }
#if TEST_STD_VER > 17
    {
        testbuf<wchar_t> sb(L"   abcdefghijk");
        std::wistream is(&sb);
        wchar_t s[4];
        is >> s;
        assert(!is.eof());
        assert(!is.fail());
        assert(std::wstring(s) == L"abc");
    }
#endif
#endif // TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<char> sb("   abcdefghijk");
        std::istream is(&sb);
        char s[20];
        is.width(1);
        is >> s;
        assert(!is.eof());
        assert( is.fail());
        assert(std::string(s) == "");
        assert(is.width() == 0);
    }
#if TEST_STD_VER > 17
    {
        testbuf<char> sb("   abcdefghijk");
        std::istream is(&sb);
        char s[1];
        is >> s;
        assert(!is.eof());
        assert( is.fail());
        assert(std::string(s) == "");
    }
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        testbuf<char> sb;
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios_base::failbit);

        bool threw = false;
        try {
            char s[20];
            is.width(10);
            is >> s;
        } catch (std::ios_base::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(is.fail());
        assert(is.eof());
        assert(threw);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb;
        std::wistream is(&sb);
        is.exceptions(std::ios_base::failbit);

        bool threw = false;
        try {
            wchar_t s[20];
            is.width(10);
            is >> s;
        } catch (std::ios_base::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(is.fail());
        assert(is.eof());
        assert(threw);
    }
#endif
    {
        testbuf<char> sb;
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios_base::eofbit);

        bool threw = false;
        try {
            char s[20];
            is.width(10);
            is >> s;
        } catch (std::ios_base::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(is.fail());
        assert(is.eof());
        assert(threw);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb;
        std::wistream is(&sb);
        is.exceptions(std::ios_base::eofbit);

        bool threw = false;
        try {
            wchar_t s[20];
            is.width(10);
            is >> s;
        } catch (std::ios_base::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(is.fail());
        assert(is.eof());
        assert(threw);
    }
#endif
#endif // TEST_HAS_NO_EXCEPTIONS

    return 0;
}
