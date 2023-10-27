//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

// <istream>

// int_type get();

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
        testbuf<char> sb("          ");
        std::istream is(&sb);
        char c = static_cast<char>(is.get());
        assert(!is.eof());
        assert(!is.fail());
        assert(c == ' ');
        assert(is.gcount() == 1);
    }
    {
        testbuf<char> sb(" abc");
        std::istream is(&sb);
        char c = static_cast<char>(is.get());
        assert(!is.eof());
        assert(!is.fail());
        assert(c == ' ');
        assert(is.gcount() == 1);
        c = static_cast<char>(is.get());
        assert(!is.eof());
        assert(!is.fail());
        assert(c == 'a');
        assert(is.gcount() == 1);
        c = static_cast<char>(is.get());
        assert(!is.eof());
        assert(!is.fail());
        assert(c == 'b');
        assert(is.gcount() == 1);
        c = static_cast<char>(is.get());
        assert(!is.eof());
        assert(!is.fail());
        assert(c == 'c');
        assert(is.gcount() == 1);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L" abc");
        std::wistream is(&sb);
        wchar_t c = is.get();
        assert(!is.eof());
        assert(!is.fail());
        assert(c == L' ');
        assert(is.gcount() == 1);
        c = is.get();
        assert(!is.eof());
        assert(!is.fail());
        assert(c == L'a');
        assert(is.gcount() == 1);
        c = is.get();
        assert(!is.eof());
        assert(!is.fail());
        assert(c == L'b');
        assert(is.gcount() == 1);
        c = is.get();
        assert(!is.eof());
        assert(!is.fail());
        assert(c == L'c');
        assert(is.gcount() == 1);
    }
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        testbuf<char> sb("rrrrrrrrr");
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios_base::eofbit);

        bool threw = false;
        try {
            while (true) {
                is.get();
                if (is.eof())
                    break;
            }
        } catch (std::ios_base::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert( is.fail());
        assert( is.eof());
        assert(threw);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L"rrrrrrrrr");
        std::basic_istream<wchar_t> is(&sb);
        is.exceptions(std::ios_base::eofbit);

        bool threw = false;
        try {
            while (true) {
                is.get();
                if (is.eof())
                    break;
            }
        } catch (std::ios_base::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert( is.fail());
        assert( is.eof());
        assert(threw);
    }
#endif
#endif // TEST_HAS_NO_EXCEPTIONS

    return 0;
}
