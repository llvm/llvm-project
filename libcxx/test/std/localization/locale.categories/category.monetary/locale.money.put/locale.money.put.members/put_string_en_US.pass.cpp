//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class money_put<charT, OutputIterator>

// iter_type put(iter_type s, bool intl, ios_base& f, char_type fill,
//               const string_type& units) const;

// REQUIRES: locale.en_US.UTF-8

// XFAIL: LIBCXX-WINDOWS-FIXME

#include <locale>
#include <ios>
#include <streambuf>
#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

#include "platform_support.h" // locale name macros

typedef std::money_put<char, cpp17_output_iterator<char*> > Fn;

class my_facet
    : public Fn
{
public:
    explicit my_facet(std::size_t refs = 0)
        : Fn(refs) {}
};

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
typedef std::money_put<wchar_t, cpp17_output_iterator<wchar_t*> > Fw;

class my_facetw
    : public Fw
{
public:
    explicit my_facetw(std::size_t refs = 0)
        : Fw(refs) {}
};
#endif

int main(int, char**)
{
    std::ios ios(0);
    std::string loc_name(LOCALE_en_US_UTF_8);
    ios.imbue(std::locale(ios.getloc(),
                          new std::moneypunct_byname<char, false>(loc_name)));
    ios.imbue(std::locale(ios.getloc(),
                          new std::moneypunct_byname<char, true>(loc_name)));
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    ios.imbue(std::locale(ios.getloc(),
                          new std::moneypunct_byname<wchar_t, false>(loc_name)));
    ios.imbue(std::locale(ios.getloc(),
                          new std::moneypunct_byname<wchar_t, true>(loc_name)));
#endif
{
    const my_facet f(1);
    // char, national
    {   // zero
        std::string v = "0";
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), false, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "0.00");
    }
    {   // negative one
        std::string v = "-1";
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), false, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "-0.01");
    }
    {   // positive
        std::string v = "123456789";
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), false, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "1,234,567.89");
    }
    {   // negative
        std::string v = "-123456789";
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), false, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "-1,234,567.89");
    }
    {   // zero, showbase
        std::string v = "0";
        std::showbase(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), false, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "$0.00");
    }
    {   // negative one, showbase
        std::string v = "-1";
        std::showbase(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), false, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "-$0.01");
    }
    {   // positive, showbase
        std::string v = "123456789";
        std::showbase(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), false, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "$1,234,567.89");
    }
    {   // negative, showbase
        std::string v = "-123456789";
        std::showbase(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), false, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "-$1,234,567.89");
    }
    {   // negative, showbase, left
        std::string v = "-123456789";
        std::showbase(ios);
        ios.width(20);
        std::left(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), false, ios, ' ', v);
        std::string ex(str, iter.base());
        assert(ex == "-$1,234,567.89      ");
        assert(ios.width() == 0);
    }
    {   // negative, showbase, internal
        std::string v = "-123456789";
        std::showbase(ios);
        ios.width(20);
        std::internal(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), false, ios, ' ', v);
        std::string ex(str, iter.base());
        assert(ex == "-$      1,234,567.89");
        assert(ios.width() == 0);
    }
    {   // negative, showbase, right
        std::string v = "-123456789";
        std::showbase(ios);
        ios.width(20);
        std::right(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), false, ios, ' ', v);
        std::string ex(str, iter.base());
        assert(ex == "      -$1,234,567.89");
        assert(ios.width() == 0);
    }

    // char, international
    std::noshowbase(ios);
    ios.unsetf(std::ios_base::adjustfield);
    {   // zero
        std::string v = "0";
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), true, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "0.00");
    }
    {   // negative one
        std::string v = "-1";
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), true, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "-0.01");
    }
    {   // positive
        std::string v = "123456789";
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), true, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "1,234,567.89");
    }
    {   // negative
        std::string v = "-123456789";
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), true, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "-1,234,567.89");
    }
    {   // zero, showbase
        std::string v = "0";
        std::showbase(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), true, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "USD 0.00");
    }
    {   // negative one, showbase
        std::string v = "-1";
        std::showbase(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), true, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "-USD 0.01");
    }
    {   // positive, showbase
        std::string v = "123456789";
        std::showbase(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), true, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "USD 1,234,567.89");
    }
    {   // negative, showbase
        std::string v = "-123456789";
        std::showbase(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), true, ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "-USD 1,234,567.89");
    }
    {   // negative, showbase, left
        std::string v = "-123456789";
        std::showbase(ios);
        ios.width(20);
        std::left(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), true, ios, ' ', v);
        std::string ex(str, iter.base());
        assert(ex == "-USD 1,234,567.89   ");
        assert(ios.width() == 0);
    }
    {   // negative, showbase, internal
        std::string v = "-123456789";
        std::showbase(ios);
        ios.width(20);
        std::internal(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), true, ios, ' ', v);
        std::string ex(str, iter.base());
        assert(ex == "-USD    1,234,567.89");
        assert(ios.width() == 0);
    }
    {   // negative, showbase, right
        std::string v = "-123456789";
        std::showbase(ios);
        ios.width(20);
        std::right(ios);
        char str[100];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), true, ios, ' ', v);
        std::string ex(str, iter.base());
        assert(ex == "   -USD 1,234,567.89");
        assert(ios.width() == 0);
    }
}
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
{
    const my_facetw f(1);
    // wchar_t, national
    std::noshowbase(ios);
    ios.unsetf(std::ios_base::adjustfield);
    {   // zero
        std::wstring v = L"0";
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), false, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"0.00");
    }
    {   // negative one
        std::wstring v = L"-1";
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), false, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"-0.01");
    }
    {   // positive
        std::wstring v = L"123456789";
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), false, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"1,234,567.89");
    }
    {   // negative
        std::wstring v = L"-123456789";
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), false, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"-1,234,567.89");
    }
    {   // zero, showbase
        std::wstring v = L"0";
        std::showbase(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), false, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"$0.00");
    }
    {   // negative one, showbase
        std::wstring v = L"-1";
        std::showbase(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), false, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"-$0.01");
    }
    {   // positive, showbase
        std::wstring v = L"123456789";
        std::showbase(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), false, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"$1,234,567.89");
    }
    {   // negative, showbase
        std::wstring v = L"-123456789";
        std::showbase(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), false, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"-$1,234,567.89");
    }
    {   // negative, showbase, left
        std::wstring v = L"-123456789";
        std::showbase(ios);
        ios.width(20);
        std::left(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), false, ios, ' ', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"-$1,234,567.89      ");
        assert(ios.width() == 0);
    }
    {   // negative, showbase, internal
        std::wstring v = L"-123456789";
        std::showbase(ios);
        ios.width(20);
        std::internal(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), false, ios, ' ', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"-$      1,234,567.89");
        assert(ios.width() == 0);
    }
    {   // negative, showbase, right
        std::wstring v = L"-123456789";
        std::showbase(ios);
        ios.width(20);
        std::right(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), false, ios, ' ', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"      -$1,234,567.89");
        assert(ios.width() == 0);
    }

    // wchar_t, international
    std::noshowbase(ios);
    ios.unsetf(std::ios_base::adjustfield);
    {   // zero
        std::wstring v = L"0";
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), true, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"0.00");
    }
    {   // negative one
        std::wstring v = L"-1";
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), true, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"-0.01");
    }
    {   // positive
        std::wstring v = L"123456789";
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), true, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"1,234,567.89");
    }
    {   // negative
        std::wstring v = L"-123456789";
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), true, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"-1,234,567.89");
    }
    {   // zero, showbase
        std::wstring v = L"0";
        std::showbase(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), true, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"USD 0.00");
    }
    {   // negative one, showbase
        std::wstring v = L"-1";
        std::showbase(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), true, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"-USD 0.01");
    }
    {   // positive, showbase
        std::wstring v = L"123456789";
        std::showbase(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), true, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"USD 1,234,567.89");
    }
    {   // negative, showbase
        std::wstring v = L"-123456789";
        std::showbase(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), true, ios, '*', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"-USD 1,234,567.89");
    }
    {   // negative, showbase, left
        std::wstring v = L"-123456789";
        std::showbase(ios);
        ios.width(20);
        std::left(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), true, ios, ' ', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"-USD 1,234,567.89   ");
        assert(ios.width() == 0);
    }
    {   // negative, showbase, internal
        std::wstring v = L"-123456789";
        std::showbase(ios);
        ios.width(20);
        std::internal(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), true, ios, ' ', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"-USD    1,234,567.89");
        assert(ios.width() == 0);
    }
    {   // negative, showbase, right
        std::wstring v = L"-123456789";
        std::showbase(ios);
        ios.width(20);
        std::right(ios);
        wchar_t str[100];
        cpp17_output_iterator<wchar_t*> iter = f.put(cpp17_output_iterator<wchar_t*>(str), true, ios, ' ', v);
        std::wstring ex(str, iter.base());
        assert(ex == L"   -USD 1,234,567.89");
        assert(ios.width() == 0);
    }
}
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
