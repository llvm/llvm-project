//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT> struct regex_traits;

// charT translate_nocase(charT c) const;

// REQUIRES: locale.en_US.UTF-8
// XFAIL: broken-utf8-wchar-ctype

#include <regex>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h"

int main(int, char**)
{
    {
        std::regex_traits<char> t;
        assert(t.translate_nocase(' ') == ' ');
        assert(t.translate_nocase('A') == 'a');
        assert(t.translate_nocase('\x07') == '\x07');
        assert(t.translate_nocase('.') == '.');
        assert(t.translate_nocase('a') == 'a');
        assert(t.translate_nocase('1') == '1');
        assert(t.translate_nocase('\xDA') == '\xDA');
        assert(t.translate_nocase('\xFA') == '\xFA');
        t.imbue(std::locale(LOCALE_en_US_UTF_8));
        assert(t.translate_nocase(' ') == ' ');
        assert(t.translate_nocase('A') == 'a');
        assert(t.translate_nocase('\x07') == '\x07');
        assert(t.translate_nocase('.') == '.');
        assert(t.translate_nocase('a') == 'a');
        assert(t.translate_nocase('1') == '1');
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::regex_traits<wchar_t> t;
        assert(t.translate_nocase(L' ') == L' ');
        assert(t.translate_nocase(L'A') == L'a');
        assert(t.translate_nocase(L'\x07') == L'\x07');
        assert(t.translate_nocase(L'.') == L'.');
        assert(t.translate_nocase(L'a') == L'a');
        assert(t.translate_nocase(L'1') == L'1');
        assert(t.translate_nocase(L'\xDA') == L'\xDA');
        assert(t.translate_nocase(L'\xFA') == L'\xFA');
        t.imbue(std::locale(LOCALE_en_US_UTF_8));
        assert(t.translate_nocase(L' ') == L' ');
        assert(t.translate_nocase(L'A') == L'a');
        assert(t.translate_nocase(L'\x07') == L'\x07');
        assert(t.translate_nocase(L'.') == L'.');
        assert(t.translate_nocase(L'a') == L'a');
        assert(t.translate_nocase(L'1') == L'1');
        assert(t.translate_nocase(L'\xDA') == L'\xFA');
        assert(t.translate_nocase(L'\xFA') == L'\xFA');
    }
#endif

  return 0;
}
