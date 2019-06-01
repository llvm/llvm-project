//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8

// <locale>

// template <class charT> class ctype_byname;

// const char* widen(const char* low, const char* high, charT* to) const;

// I doubt this test is portable

#include <locale>
#include <string>
#include <vector>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

int main(int, char**)
{
    {
        std::locale l(LOCALE_en_US_UTF_8);
        {
            typedef std::ctype_byname<wchar_t> F;
            std::locale ll(l, new F(LOCALE_en_US_UTF_8));
            F const& f = std::use_facet<F>(ll);
            std::string in(" A\x07.a1\x85");
            std::vector<wchar_t> v(in.size());

            assert(f.widen(&in[0], in.data() + in.size(), v.data()) == in.data() + in.size());
            assert(v[0] == L' ');
            assert(v[1] == L'A');
            assert(v[2] == L'\x07');
            assert(v[3] == L'.');
            assert(v[4] == L'a');
            assert(v[5] == L'1');
            assert(v[6] == wchar_t(-1));
        }
    }
    {
        std::locale l("C");
        {
            typedef std::ctype_byname<wchar_t> F;
            std::locale ll(l, new F("C"));
            const F& f = std::use_facet<F>(ll);
            std::string in(" A\x07.a1\x85");
            std::vector<wchar_t> v(in.size());

            assert(f.widen(&in[0], in.data() + in.size(), v.data()) == in.data() + in.size());
            assert(v[0] == L' ');
            assert(v[1] == L'A');
            assert(v[2] == L'\x07');
            assert(v[3] == L'.');
            assert(v[4] == L'a');
            assert(v[5] == L'1');
#if defined(__APPLE__) || defined(__FreeBSD__)
            assert(v[6] == L'\x85');
#else
            assert(v[6] == wchar_t(-1));
#endif
        }
    }

  return 0;
}
