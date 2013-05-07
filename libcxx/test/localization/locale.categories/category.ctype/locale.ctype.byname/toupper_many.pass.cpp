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

// const charT* toupper(charT* low, const charT* high) const;

#include <locale>
#include <string>
#include <cassert>

#include "platform_support.h" // locale name macros

int main()
{
    {
        std::locale l(LOCALE_en_US_UTF_8);
        {
            typedef std::ctype<char> F;
            const F& f = std::use_facet<F>(l);
            std::string in("\xFA A\x07.a1");

            assert(f.toupper(&in[0], in.data() + in.size()) == in.data() + in.size());
            assert(in[0] == '\xDA');
            assert(in[1] == ' ');
            assert(in[2] == 'A');
            assert(in[3] == '\x07');
            assert(in[4] == '.');
            assert(in[5] == 'A');
            assert(in[6] == '1');
        }
    }
    {
        std::locale l("C");
        {
            typedef std::ctype<char> F;
            const F& f = std::use_facet<F>(l);
            std::string in("\xFA A\x07.a1");

            assert(f.toupper(&in[0], in.data() + in.size()) == in.data() + in.size());
            assert(in[0] == '\xFA');
            assert(in[1] == ' ');
            assert(in[2] == 'A');
            assert(in[3] == '\x07');
            assert(in[4] == '.');
            assert(in[5] == 'A');
            assert(in[6] == '1');
        }
    }
    {
        std::locale l(LOCALE_en_US_UTF_8);
        {
            typedef std::ctype<wchar_t> F;
            const F& f = std::use_facet<F>(l);
            std::wstring in(L"\xFA A\x07.a1");

            assert(f.toupper(&in[0], in.data() + in.size()) == in.data() + in.size());
            assert(in[0] == L'\xDA');
            assert(in[1] == L' ');
            assert(in[2] == L'A');
            assert(in[3] == L'\x07');
            assert(in[4] == L'.');
            assert(in[5] == L'A');
            assert(in[6] == L'1');
        }
    }
    {
        std::locale l("C");
        {
            typedef std::ctype<wchar_t> F;
            const F& f = std::use_facet<F>(l);
            std::wstring in(L"\xFA A\x07.a1");

            assert(f.toupper(&in[0], in.data() + in.size()) == in.data() + in.size());
            assert(in[0] == L'\xFA');
            assert(in[1] == L' ');
            assert(in[2] == L'A');
            assert(in[3] == L'\x07');
            assert(in[4] == L'.');
            assert(in[5] == L'A');
            assert(in[6] == L'1');
        }
    }
}
