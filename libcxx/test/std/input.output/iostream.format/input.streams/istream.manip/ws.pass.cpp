//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <istream>

// template <class charT, class traits>
//   basic_istream<charT,traits>&
//   ws(basic_istream<charT,traits>& is);

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
        testbuf<char> sb("   123");
        std::istream is(&sb);
        std::ws(is);
        assert(is.good());
        assert(is.peek() == '1');
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L"   123");
        std::wistream is(&sb);
        std::ws(is);
        assert(is.good());
        assert(is.peek() == L'1');
    }
#endif
    {
        testbuf<char> sb("  ");
        std::istream is(&sb);
        std::ws(is);
        assert(!is.fail());
        assert(is.eof());
        std::ws(is);
        assert(is.eof());
        assert(is.fail());
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L"  ");
        std::wistream is(&sb);
        std::ws(is);
        assert(!is.fail());
        assert(is.eof());
        std::ws(is);
        assert(is.eof());
        assert(is.fail());
    }
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        testbuf<char> sb("  ");
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios_base::eofbit);

        bool threw = false;
        try {
            std::ws(is);
        } catch (std::ios_base::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(!is.fail());
        assert( is.eof());
        assert(threw);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L"  ");
        std::basic_istream<wchar_t> is(&sb);
        is.exceptions(std::ios_base::eofbit);

        bool threw = false;
        try {
            std::ws(is);
        } catch (std::ios_base::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(!is.fail());
        assert( is.eof());
        assert(threw);
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS
#endif // TEST_HAS_NO_EXCEPTIONS

    return 0;
}
