//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class ctype_byname;

// charT widen(char c) const;

// I doubt this test is portable

#include <locale>
#include <cassert>
#include <limits.h>

#include "../../../../platform_support.h" // locale name macros

int main()
{
    {
        std::locale l(LOCALE_en_US_UTF_8);
        {
            typedef std::ctype<wchar_t> F;
            const F& f = std::use_facet<F>(l);

            assert(f.widen(' ') == L' ');
            assert(f.widen('A') == L'A');
            assert(f.widen('\x07') == L'\x07');
            assert(f.widen('.') == L'.');
            assert(f.widen('a') == L'a');
            assert(f.widen('1') == L'1');
            assert(f.widen(char(-5)) == wchar_t(-1));
        }
    }
    {
        std::locale l("C");
        {
            typedef std::ctype<wchar_t> F;
            const F& f = std::use_facet<F>(l);

            assert(f.widen(' ') == L' ');
            assert(f.widen('A') == L'A');
            assert(f.widen('\x07') == L'\x07');
            assert(f.widen('.') == L'.');
            assert(f.widen('a') == L'a');
            assert(f.widen('1') == L'1');
            assert(f.widen(char(-5)) == wchar_t(251));
        }
    }
}
