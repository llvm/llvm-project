//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8
// XFAIL: win32-broken-utf8-wchar-ctype

// Bionic has minimal locale support, investigate this later.
// XFAIL: LIBCXX-ANDROID-FIXME

// <locale>

// template <class charT> class ctype_byname;

// charT toupper(charT) const;


#include <locale>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

int main(int, char**)
{
    {
        std::locale l;
        {
            typedef std::ctype_byname<char> F;
            std::locale ll(l, new F(LOCALE_en_US_UTF_8));
            const F& f = std::use_facet<F>(ll);

            assert(f.toupper(' ') == ' ');
            assert(f.toupper('A') == 'A');
            assert(f.toupper('\x07') == '\x07');
            assert(f.toupper('.') == '.');
            assert(f.toupper('a') == 'A');
            assert(f.toupper('1') == '1');
            assert(f.toupper('\xDA') == '\xDA');
            assert(f.toupper('c') == 'C');
        }
    }
    {
        std::locale l;
        {
            typedef std::ctype_byname<char> F;
            std::locale ll(l, new F("C"));
            const F& f = std::use_facet<F>(ll);

            assert(f.toupper(' ') == ' ');
            assert(f.toupper('A') == 'A');
            assert(f.toupper('\x07') == '\x07');
            assert(f.toupper('.') == '.');
            assert(f.toupper('a') == 'A');
            assert(f.toupper('1') == '1');
            assert(f.toupper('\xDA') == '\xDA');
            assert(f.toupper('\xFA') == '\xFA');
        }
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::locale l;
        {
            typedef std::ctype_byname<wchar_t> F;
            std::locale ll(l, new F(LOCALE_en_US_UTF_8));
            const F& f = std::use_facet<F>(ll);

            assert(f.toupper(L' ') == L' ');
            assert(f.toupper(L'A') == L'A');
            assert(f.toupper(L'\x07') == L'\x07');
            assert(f.toupper(L'.') == L'.');
            assert(f.toupper(L'a') == L'A');
            assert(f.toupper(L'1') == L'1');
            assert(f.toupper(L'\xDA') == L'\xDA');
            assert(f.toupper(L'\xFA') == L'\xDA');
        }
    }
    {
        std::locale l;
        {
            typedef std::ctype_byname<wchar_t> F;
            std::locale ll(l, new F("C"));
            const F& f = std::use_facet<F>(ll);

            assert(f.toupper(L' ') == L' ');
            assert(f.toupper(L'A') == L'A');
            assert(f.toupper(L'\x07') == L'\x07');
            assert(f.toupper(L'.') == L'.');
            assert(f.toupper(L'a') == L'A');
            assert(f.toupper(L'1') == L'1');
            assert(f.toupper(L'\xDA') == L'\xDA');
            assert(f.toupper(L'\xFA') == L'\xFA');
        }
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
