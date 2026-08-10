//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_CA.ISO8859-1
// XFAIL: no-wide-characters

// <locale>

// template <class charT> class ctype_byname;

// char narrow(charT c, char dfault) const;

#include <locale>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

int main(int, char**)
{
    {
        std::locale l(std::string(LOCALE_fr_CA_ISO8859_1));
        {
            typedef std::ctype<wchar_t> F;
            const F& f = std::use_facet<F>(l);

            assert(f.narrow(L' ', '*') == ' ');
            assert(f.narrow(L'A', '*') == 'A');
            assert(f.narrow(L'\x07', '*') == '\x07');
            assert(f.narrow(L'.', '*') == '.');
            assert(f.narrow(L'a', '*') == 'a');
            assert(f.narrow(L'1', '*') == '1');
            assert(f.narrow(L'\xDA', '*') == '\xDA');
        }
    }
    {
        std::locale l(LOCALE_en_US_UTF_8);
        {
            typedef std::ctype<wchar_t> F;
            const F& f = std::use_facet<F>(l);

            assert(f.narrow(L' ', '*') == ' ');
            assert(f.narrow(L'A', '*') == 'A');
            assert(f.narrow(L'\x07', '*') == '\x07');
            assert(f.narrow(L'.', '*') == '.');
            assert(f.narrow(L'a', '*') == 'a');
            assert(f.narrow(L'1', '*') == '1');
            assert(f.narrow(L'\xDA', '*') == '*');
        }
    }

  return 0;
}
