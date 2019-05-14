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
//   operator>>(basic_istream<charT,traits>& is,
//              basic_string<charT,traits,Allocator>& str);

#include <string>
#include <sstream>
#include <cassert>

#include "min_allocator.h"
#include "test_macros.h"

int main(int, char**)
{
    {
        std::istringstream in("a bc defghij");
        std::string s("initial text");
        in >> s;
        assert(in.good());
        assert(s == "a");
        assert(in.peek() == ' ');
        in >> s;
        assert(in.good());
        assert(s == "bc");
        assert(in.peek() == ' ');
        in.width(3);
        in >> s;
        assert(in.good());
        assert(s == "def");
        assert(in.peek() == 'g');
        in >> s;
        assert(in.eof());
        assert(s == "ghij");
        in >> s;
        assert(in.fail());
    }
    {
        std::wistringstream in(L"a bc defghij");
        std::wstring s(L"initial text");
        in >> s;
        assert(in.good());
        assert(s == L"a");
        assert(in.peek() == L' ');
        in >> s;
        assert(in.good());
        assert(s == L"bc");
        assert(in.peek() == L' ');
        in.width(3);
        in >> s;
        assert(in.good());
        assert(s == L"def");
        assert(in.peek() == L'g');
        in >> s;
        assert(in.eof());
        assert(s == L"ghij");
        in >> s;
        assert(in.fail());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::stringbuf sb;
        std::istream is(&sb);
        is.exceptions(std::ios::failbit);

        bool threw = false;
        try {
            std::string s;
            is >> s;
        } catch (std::ios::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(is.fail());
        assert(is.eof());
        assert(threw);
    }
    {
        std::stringbuf sb;
        std::istream is(&sb);
        is.exceptions(std::ios::eofbit);

        bool threw = false;
        try {
            std::string s;
            is >> s;
        } catch (std::ios::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(is.fail());
        assert(is.eof());
        assert(threw);
    }
#endif // TEST_HAS_NO_EXCEPTIONS
#if TEST_STD_VER >= 11
    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
        std::istringstream in("a bc defghij");
        S s("initial text");
        in >> s;
        assert(in.good());
        assert(s == "a");
        assert(in.peek() == ' ');
        in >> s;
        assert(in.good());
        assert(s == "bc");
        assert(in.peek() == ' ');
        in.width(3);
        in >> s;
        assert(in.good());
        assert(s == "def");
        assert(in.peek() == 'g');
        in >> s;
        assert(in.eof());
        assert(s == "ghij");
        in >> s;
        assert(in.fail());
    }
    {
        typedef std::basic_string<wchar_t, std::char_traits<wchar_t>, min_allocator<wchar_t>> S;
        std::wistringstream in(L"a bc defghij");
        S s(L"initial text");
        in >> s;
        assert(in.good());
        assert(s == L"a");
        assert(in.peek() == L' ');
        in >> s;
        assert(in.good());
        assert(s == L"bc");
        assert(in.peek() == L' ');
        in.width(3);
        in >> s;
        assert(in.good());
        assert(s == L"def");
        assert(in.peek() == L'g');
        in >> s;
        assert(in.eof());
        assert(s == L"ghij");
        in >> s;
        assert(in.fail());
    }
#endif

  return 0;
}
