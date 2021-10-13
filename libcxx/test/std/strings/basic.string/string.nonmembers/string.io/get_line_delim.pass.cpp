//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   basic_istream<charT,traits>&
//   getline(basic_istream<charT,traits>& is,
//           basic_string<charT,traits,Allocator>& str, charT delim);

#include <string>
#include <sstream>
#include <cassert>

#include "min_allocator.h"
#include "test_macros.h"

int main(int, char**)
{
    {
        std::istringstream in(" abc*  def**   ghij");
        std::string s("initial text");
        getline(in, s, '*');
        assert(in.good());
        assert(s == " abc");
        getline(in, s, '*');
        assert(in.good());
        assert(s == "  def");
        getline(in, s, '*');
        assert(in.good());
        assert(s == "");
        getline(in, s, '*');
        assert(in.eof());
        assert(s == "   ghij");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wistringstream in(L" abc*  def**   ghij");
        std::wstring s(L"initial text");
        getline(in, s, L'*');
        assert(in.good());
        assert(s == L" abc");
        getline(in, s, L'*');
        assert(in.good());
        assert(s == L"  def");
        getline(in, s, L'*');
        assert(in.good());
        assert(s == L"");
        getline(in, s, L'*');
        assert(in.eof());
        assert(s == L"   ghij");
    }
#endif
#if TEST_STD_VER >= 11
    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
        std::istringstream in(" abc*  def**   ghij");
        S s("initial text");
        getline(in, s, '*');
        assert(in.good());
        assert(s == " abc");
        getline(in, s, '*');
        assert(in.good());
        assert(s == "  def");
        getline(in, s, '*');
        assert(in.good());
        assert(s == "");
        getline(in, s, '*');
        assert(in.eof());
        assert(s == "   ghij");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        typedef std::basic_string<wchar_t, std::char_traits<wchar_t>, min_allocator<wchar_t>> S;
        std::wistringstream in(L" abc*  def**   ghij");
        S s(L"initial text");
        getline(in, s, L'*');
        assert(in.good());
        assert(s == L" abc");
        getline(in, s, L'*');
        assert(in.good());
        assert(s == L"  def");
        getline(in, s, L'*');
        assert(in.good());
        assert(s == L"");
        getline(in, s, L'*');
        assert(in.eof());
        assert(s == L"   ghij");
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS
#endif // TEST_STD_VER >= 11
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::basic_stringbuf<char> sb("hello");
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios::eofbit);

        std::basic_string<char> s;
        bool threw = false;
        try {
            std::getline(is, s, '\n');
        } catch (std::ios::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(!is.fail());
        assert( is.eof());
        assert(threw);
        assert(s == "hello");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::basic_stringbuf<wchar_t> sb(L"hello");
        std::basic_istream<wchar_t> is(&sb);
        is.exceptions(std::ios::eofbit);

        std::basic_string<wchar_t> s;
        bool threw = false;
        try {
            std::getline(is, s, L'\n');
        } catch (std::ios::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(!is.fail());
        assert( is.eof());
        assert(threw);
        assert(s == L"hello");
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::basic_stringbuf<char> sb;
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios::failbit);

        std::basic_string<char> s;
        bool threw = false;
        try {
            std::getline(is, s, '\n');
        } catch (std::ios::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert( is.fail());
        assert( is.eof());
        assert(threw);
        assert(s == "");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::basic_stringbuf<wchar_t> sb;
        std::basic_istream<wchar_t> is(&sb);
        is.exceptions(std::ios::failbit);

        std::basic_string<wchar_t> s;
        bool threw = false;
        try {
            std::getline(is, s, L'\n');
        } catch (std::ios::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert( is.fail());
        assert( is.eof());
        assert(threw);
        assert(s == L"");
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS
#endif // TEST_HAS_NO_EXCEPTIONS

    return 0;
}
