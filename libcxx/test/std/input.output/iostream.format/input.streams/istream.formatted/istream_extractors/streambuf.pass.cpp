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

// template <class charT, class traits = char_traits<charT> >
//   class basic_istream;

// basic_istream<charT,traits>& operator>>(basic_streambuf<charT,traits>* sb);

#include <istream>
#include <cassert>

#include "test_macros.h"

template <class CharT>
class testbuf
    : public std::basic_streambuf<CharT>
{
    typedef std::basic_streambuf<CharT> base;
    std::basic_string<CharT> str_;
public:
    testbuf()
    {
    }
    testbuf(const std::basic_string<CharT>& str)
        : str_(str)
    {
        base::setg(const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data() + str_.size()));
    }

    std::basic_string<CharT> str() const
        {return std::basic_string<CharT>(base::pbase(), base::pptr());}

protected:

    virtual typename base::int_type
        overflow(typename base::int_type ch = base::traits_type::eof())
        {
            if (ch != base::traits_type::eof())
            {
                std::size_t n = str_.size();
                str_.push_back(static_cast<CharT>(ch));
                str_.resize(str_.capacity());
                base::setp(const_cast<CharT*>(str_.data()),
                           const_cast<CharT*>(str_.data() + str_.size()));
                base::pbump(static_cast<int>(n+1));
            }
            return ch;
        }
};

int main(int, char**)
{
    {
        testbuf<char> sb("testing...");
        std::istream is(&sb);
        testbuf<char> sb2;
        is >> &sb2;
        assert(sb2.str() == "testing...");
        assert(is.gcount() == 10);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        testbuf<char> sb(" ");
        std::basic_istream<char> is(&sb);
        testbuf<char> sb2;
        is.exceptions(std::istream::eofbit);
        bool threw = false;
        try {
            is >> &sb2;
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
        testbuf<wchar_t> sb2;
        is.exceptions(std::istream::eofbit);
        bool threw = false;
        try {
            is >> &sb2;
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(threw);
        assert(!is.bad());
        assert( is.eof());
        assert(!is.fail());
    }
#endif

    {
        testbuf<char> sb;
        std::basic_istream<char> is(&sb);
        testbuf<char> sb2;
        is.exceptions(std::istream::failbit);
        bool threw = false;
        try {
            is >> &sb2;
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(threw);
        assert(!is.bad());
        assert( is.eof());
        assert( is.fail());
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb;
        std::basic_istream<wchar_t> is(&sb);
        testbuf<wchar_t> sb2;
        is.exceptions(std::istream::failbit);
        bool threw = false;
        try {
            is >> &sb2;
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(threw);
        assert(!is.bad());
        assert( is.eof());
        assert( is.fail());
    }
#endif

    {
        testbuf<char> sb;
        std::basic_istream<char> is(&sb);
        is.exceptions(std::istream::failbit);
        bool threw = false;
        try {
            is >> static_cast<testbuf<char>*>(0);
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(threw);
        assert(!is.bad());
        assert(!is.eof());
        assert( is.fail());
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb;
        std::basic_istream<wchar_t> is(&sb);
        is.exceptions(std::istream::failbit);
        bool threw = false;
        try {
            is >> static_cast<testbuf<wchar_t>*>(0);
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(threw);
        assert(!is.bad());
        assert(!is.eof());
        assert( is.fail());
    }
#endif
#endif // TEST_HAS_NO_EXCEPTIONS

    return 0;
}
