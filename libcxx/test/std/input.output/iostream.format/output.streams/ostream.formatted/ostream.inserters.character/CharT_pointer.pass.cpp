//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ostream>

// template <class charT, class traits = char_traits<charT> >
//   class basic_ostream;

// template<class charT, class traits>
//   basic_ostream<charT,traits>& operator<<(basic_ostream<charT,traits>& out, const charT* s);

// XFAIL: libcpp-has-no-wide-characters

#include <ostream>
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

    std::basic_string<CharT> str() const
        {return std::basic_string<CharT>(base::pbase(), base::pptr());}

protected:

    virtual typename base::int_type
        overflow(typename base::int_type ch = base::traits_type::eof())
        {
            if (ch != base::traits_type::eof())
            {
                int n = static_cast<int>(str_.size());
                str_.push_back(ch);
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
        std::wostream os((std::wstreambuf*)0);
        const wchar_t* c = L"123";
        os << c;
        assert(os.bad());
        assert(os.fail());
    }
    {
        testbuf<wchar_t> sb;
        std::wostream os(&sb);
        const wchar_t* c = L"123";
        os << c;
        assert(sb.str() == L"123");
    }
    {
        testbuf<wchar_t> sb;
        std::wostream os(&sb);
        os.width(5);
        const wchar_t* c = L"123";
        os << c;
        assert(sb.str() == L"  123");
        assert(os.width() == 0);
    }
    {
        testbuf<wchar_t> sb;
        std::wostream os(&sb);
        os.width(5);
        left(os);
        const wchar_t* c = L"123";
        os << c;
        assert(sb.str() == L"123  ");
        assert(os.width() == 0);
    }

  return 0;
}
