//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT> struct regex_traits;

// bool isctype(charT c, char_class_type f) const;


#include <regex>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
    {
        std::regex_traits<char> t;

        std::string s("w");
        assert( t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('@', t.lookup_classname(s.begin(), s.end())));

        s = "alnum";
        assert(!t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('@', t.lookup_classname(s.begin(), s.end())));

        s = "alpha";
        assert(!t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('@', t.lookup_classname(s.begin(), s.end())));

        s = "blank";
        assert(!t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('@', t.lookup_classname(s.begin(), s.end())));

        s = "cntrl";
        assert( t.isctype('\n', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('@', t.lookup_classname(s.begin(), s.end())));

        s = "digit";
        assert(!t.isctype('\n', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('@', t.lookup_classname(s.begin(), s.end())));

        s = "graph";
        assert(!t.isctype('\n', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('@', t.lookup_classname(s.begin(), s.end())));

        s = "lower";
        assert(!t.isctype('\n', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('@', t.lookup_classname(s.begin(), s.end())));

        s = "print";
        assert(!t.isctype('\n', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('@', t.lookup_classname(s.begin(), s.end())));

        s = "punct";
        assert(!t.isctype('\n', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('@', t.lookup_classname(s.begin(), s.end())));

        s = "space";
        assert( t.isctype('\n', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('@', t.lookup_classname(s.begin(), s.end())));

        s = "upper";
        assert(!t.isctype('\n', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('@', t.lookup_classname(s.begin(), s.end())));

        s = "xdigit";
        assert(!t.isctype('\n', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('Z', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype('5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype('@', t.lookup_classname(s.begin(), s.end())));
    }

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::regex_traits<wchar_t> t;

        std::wstring s(L"w");
        assert( t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));

        s = L"alnum";
        assert(!t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));

        s = L"alpha";
        assert(!t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));

        s = L"blank";
        assert(!t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));

        s = L"cntrl";
        assert( t.isctype(L'\n', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));

        s = L"digit";
        assert(!t.isctype(L'\n', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));

        s = L"graph";
        assert(!t.isctype(L'\n', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));

        s = L"lower";
        assert(!t.isctype(L'\n', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));

        s = L"print";
        assert(!t.isctype(L'\n', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));

        s = L"punct";
        assert(!t.isctype(L'\n', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));

        s = L"space";
        assert( t.isctype(L'\n', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));

        s = L"upper";
        assert(!t.isctype(L'\n', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));

        s = L"xdigit";
        assert(!t.isctype(L'\n', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'_', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'a', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'Z', t.lookup_classname(s.begin(), s.end())));
        assert( t.isctype(L'5', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L' ', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'-', t.lookup_classname(s.begin(), s.end())));
        assert(!t.isctype(L'@', t.lookup_classname(s.begin(), s.end())));
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
