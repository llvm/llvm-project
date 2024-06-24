//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: stdlib=system && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

// <istream>

// int_type peek();

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
        testbuf<char> sb(" 1\n2345\n6");
        std::istream is(&sb);
        assert(is.peek() == ' ');
        assert(!is.eof());
        assert(!is.fail());
        assert(is.gcount() == 0);
        is.get();
        assert(is.peek() == '1');
        assert(!is.eof());
        assert(!is.fail());
        assert(is.gcount() == 0);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L" 1\n2345\n6");
        std::wistream is(&sb);
        assert(is.peek() == L' ');
        assert(!is.eof());
        assert(!is.fail());
        assert(is.gcount() == 0);
        is.get();
        assert(is.peek() == L'1');
        assert(!is.eof());
        assert(!is.fail());
        assert(is.gcount() == 0);
    }
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        testbuf<char> sb;
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios_base::eofbit);
        bool threw = false;
        try {
            is.peek();
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
        testbuf<wchar_t> sb;
        std::basic_istream<wchar_t> is(&sb);
        is.exceptions(std::ios_base::eofbit);
        bool threw = false;
        try {
            is.peek();
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
