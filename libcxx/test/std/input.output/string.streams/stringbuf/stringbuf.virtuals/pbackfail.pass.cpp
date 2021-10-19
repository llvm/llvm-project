//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringbuf

// int_type pbackfail(int_type c = traits::eof());

#include <sstream>
#include <cassert>

#include "test_macros.h"

template <class CharT>
struct testbuf
    : public std::basic_stringbuf<CharT>
{
    typedef std::basic_stringbuf<CharT> base;
    explicit testbuf(const std::basic_string<CharT>& str,
                     std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
        : base(str, which) {}

    typename base::int_type
        pbackfail(typename base::int_type c = base::traits_type::eof())
        {return base::pbackfail(c);}

    void pbump(int n) {base::pbump(n);}
};

int main(int, char**)
{
    // sanity check
    {
        testbuf<char> tb("");
        tb.pbackfail();
    }
    {
        testbuf<char> sb("123", std::ios_base::in);
        assert(sb.sgetc() == '1');
        assert(sb.snextc() == '2');
        assert(sb.snextc() == '3');
        assert(sb.sgetc() == '3');
        assert(sb.snextc() == std::char_traits<char>::eof());
        assert(sb.pbackfail('3') == '3');
        assert(sb.pbackfail('3') == std::char_traits<char>::eof());
        assert(sb.pbackfail('2') == '2');
        assert(sb.pbackfail(std::char_traits<char>::eof()) != std::char_traits<char>::eof());
        assert(sb.pbackfail(std::char_traits<char>::eof()) == std::char_traits<char>::eof());
        assert(sb.str() == "123");
    }
    {
        testbuf<char> sb("123");
        assert(sb.sgetc() == '1');
        assert(sb.snextc() == '2');
        assert(sb.snextc() == '3');
        assert(sb.sgetc() == '3');
        assert(sb.snextc() == std::char_traits<char>::eof());
        assert(sb.pbackfail('3') == '3');
        assert(sb.pbackfail('3') == '3');
        assert(sb.pbackfail(std::char_traits<char>::eof()) != std::char_traits<char>::eof());
        assert(sb.pbackfail(std::char_traits<char>::eof()) == std::char_traits<char>::eof());
        assert(sb.str() == "133");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L"123", std::ios_base::in);
        assert(sb.sgetc() == L'1');
        assert(sb.snextc() == L'2');
        assert(sb.snextc() == L'3');
        assert(sb.sgetc() == L'3');
        assert(sb.snextc() == std::char_traits<wchar_t>::eof());
        assert(sb.pbackfail(L'3') == L'3');
        assert(sb.pbackfail(L'3') == std::char_traits<wchar_t>::eof());
        assert(sb.pbackfail(L'2') == L'2');
        assert(sb.pbackfail(std::char_traits<wchar_t>::eof()) != std::char_traits<wchar_t>::eof());
        assert(sb.pbackfail(std::char_traits<wchar_t>::eof()) == std::char_traits<wchar_t>::eof());
        assert(sb.str() == L"123");
    }
    {
        testbuf<wchar_t> sb(L"123");
        assert(sb.sgetc() == L'1');
        assert(sb.snextc() == L'2');
        assert(sb.snextc() == L'3');
        assert(sb.sgetc() == L'3');
        assert(sb.snextc() == std::char_traits<wchar_t>::eof());
        assert(sb.pbackfail(L'3') == L'3');
        assert(sb.pbackfail(L'3') == L'3');
        assert(sb.pbackfail(std::char_traits<wchar_t>::eof()) != std::char_traits<wchar_t>::eof());
        assert(sb.pbackfail(std::char_traits<wchar_t>::eof()) == std::char_traits<wchar_t>::eof());
        assert(sb.str() == L"133");
    }
#endif

  return 0;
}
