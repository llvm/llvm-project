//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

// <istream>

// basic_istream<charT,traits>& get(basic_streambuf<char_type,traits>& sb,
//                                  char_type delim);

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
                int n = static_cast<int>(str_.size());
                str_.push_back(static_cast<CharT>(ch));
                str_.resize(str_.capacity());
                base::setp(const_cast<CharT*>(str_.data()),
                           const_cast<CharT*>(str_.data() + str_.size()));
                base::pbump(n+1);
            }
            return ch;
        }
};

int main(int, char**)
{
    {
        testbuf<char> sb("testing*...");
        std::istream is(&sb);
        testbuf<char> sb2;
        is.get(sb2, '*');
        assert(sb2.str() == "testing");
        assert(is.good());
        assert(is.gcount() == 7);
        assert(is.get() == '*');
        is.get(sb2, '*');
        assert(sb2.str() == "testing...");
        assert(is.eof());
        assert(!is.fail());
        assert(is.gcount() == 3);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L"testing*...");
        std::wistream is(&sb);
        testbuf<wchar_t> sb2;
        is.get(sb2, L'*');
        assert(sb2.str() == L"testing");
        assert(is.good());
        assert(is.gcount() == 7);
        assert(is.get() == L'*');
        is.get(sb2, L'*');
        assert(sb2.str() == L"testing...");
        assert(is.eof());
        assert(!is.fail());
        assert(is.gcount() == 3);
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        testbuf<char> sb(" ");
        std::basic_istream<char> is(&sb);
        testbuf<char> sb2;
        is.exceptions(std::ios_base::eofbit);
        bool threw = false;
        try {
            is.get(sb2, '*');
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
        is.exceptions(std::ios_base::eofbit);
        bool threw = false;
        try {
            is.get(sb2, L'*');
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(threw);
        assert(!is.bad());
        assert( is.eof());
        assert(!is.fail());
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

    {
        testbuf<char> sb;
        std::basic_istream<char> is(&sb);
        testbuf<char> sb2;
        is.exceptions(std::ios_base::eofbit);
        bool threw = false;
        try {
            is.get(sb2, '*');
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
        is.exceptions(std::ios_base::eofbit);
        bool threw = false;
        try {
            is.get(sb2, L'*');
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(threw);
        assert(!is.bad());
        assert( is.eof());
        assert( is.fail());
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS
#endif // TEST_HAS_NO_EXCEPTIONS

    return 0;
}
