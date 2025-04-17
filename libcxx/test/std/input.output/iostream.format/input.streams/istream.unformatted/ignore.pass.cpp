//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Requires 396145d in the built library.
// XFAIL: using-built-library-before-llvm-9

// <istream>

// basic_istream<charT,traits>&
//    ignore(streamsize n = 1, int_type delim = traits::eof());

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
        testbuf<char> sb(" 1\n2345\n6");
        std::istream is(&sb);
        is.ignore();
        assert(!is.eof());
        assert(!is.fail());
        assert(is.gcount() == 1);
        is.ignore(5, '\n');
        assert(!is.eof());
        assert(!is.fail());
        assert(is.gcount() == 2);
        is.ignore(15);
        assert( is.eof());
        assert(!is.fail());
        assert(is.gcount() == 6);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L" 1\n2345\n6");
        std::wistream is(&sb);
        is.ignore();
        assert(!is.eof());
        assert(!is.fail());
        assert(is.gcount() == 1);
        is.ignore(5, '\n');
        assert(!is.eof());
        assert(!is.fail());
        assert(is.gcount() == 2);
        is.ignore(15);
        assert( is.eof());
        assert(!is.fail());
        assert(is.gcount() == 6);
    }
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        testbuf<char> sb(" ");
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios_base::eofbit);
        bool threw = false;
        try {
            is.ignore(5);
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(threw);
        assert(!is.bad());
        assert( is.eof());
        assert(!is.fail());
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L" ");
        std::basic_istream<wchar_t> is(&sb);
        is.exceptions(std::ios_base::eofbit);
        bool threw = false;
        try {
            is.ignore(5);
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(threw);
        assert(!is.bad());
        assert( is.eof());
        assert(!is.fail());
    }
#endif
#endif // TEST_HAS_NO_EXCEPTIONS

    return 0;
}
