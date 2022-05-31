//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NetBSD does not support LC_MONETARY at the moment
// XFAIL: netbsd

// REQUIRES: locale.ru_RU.UTF-8

// XFAIL: glibc-old-ru_RU-decimal-point

// <locale>

// class money_get<charT, InputIterator>

// iter_type get(iter_type b, iter_type e, bool intl, ios_base& iob,
//               ios_base::iostate& err, long double& v) const;

#include <locale>
#include <ios>
#include <streambuf>
#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

#include "locale_helpers.h"
#include "platform_support.h" // locale name macros

typedef std::money_get<char, cpp17_input_iterator<const char*> > Fn;

class my_facet
    : public Fn
{
public:
    explicit my_facet(std::size_t refs = 0)
        : Fn(refs) {}
};

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
typedef std::money_get<wchar_t, cpp17_input_iterator<const wchar_t*> > Fw;

class my_facetw
    : public Fw
{
public:
    explicit my_facetw(std::size_t refs = 0)
        : Fw(refs) {}
};

static std::wstring convert_thousands_sep(std::wstring const& in) {
  return LocaleHelpers::convert_thousands_sep_ru_RU(in);
}
#endif // TEST_HAS_NO_WIDE_CHARACTERS

int main(int, char**)
{
    std::ios ios(0);
    std::string loc_name(LOCALE_ru_RU_UTF_8);
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
    std::string symbol(LocaleHelpers::currency_symbol_ru_RU());
    {
        const my_facet f(1);
        // char, national
        {   // zero
            std::string v = "0,00 ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
        }
        {   // negative one
            std::string v = "-0,01 ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
        }
        {   // positive
            std::string v = "1 234 567,89 ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
        }
        {   // negative
            std::string v = "-1 234 567,89 ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
        }
        {   // negative
            std::string v = "-1234567,89 ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
        }
        {   // zero, showbase
            std::string v = "0,00 " + symbol;
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 5);
            assert(err == std::ios_base::goodbit);
            assert(ex == 0);
        }
        {   // zero, showbase
            std::string v = "0,00 " + symbol;
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
            std::noshowbase(ios);
        }
        {   // negative one, showbase
            std::string v = "-0,01 " + symbol;
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 6);
            assert(err == std::ios_base::goodbit);
            assert(ex == -1);
        }
        {   // negative one, showbase
            std::string v = "-0,01 " + symbol;
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
            std::noshowbase(ios);
        }
        {   // positive, showbase
            std::string v = "1 234 567,89 " + symbol;
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 13);
            assert(err == std::ios_base::goodbit);
            assert(ex == 123456789);
        }
        {   // positive, showbase
            std::string v = "1 234 567,89 " + symbol;
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = "-1 234 567,89 " + symbol;
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = "-1 234 567,89 RUB";
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 14);
            assert(err == std::ios_base::failbit);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = "-1 234 567,89 RUB";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 14);
            assert(err == std::ios_base::goodbit);
            assert(ex == -123456789);
        }
    }
    {
        const my_facet f(1);
        // char, international
        {   // zero
            std::string v = "0,00";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
        }
        {   // negative one
            std::string v = "-0,01 ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
        }
        {   // positive
            std::string v = "1 234 567,89 ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
        }
        {   // negative
            std::string v = "-1 234 567,89 ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
        }
        {   // negative
            std::string v = "-1234567,89 ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
        }
        {   // zero, showbase
            std::string v = "0,00 RUB";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 5);
            assert(err == std::ios_base::goodbit);
            assert(ex == 0);
        }
        {   // zero, showbase
            std::string v = "0,00 RUB";
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
            std::noshowbase(ios);
        }
        {   // negative one, showbase
            std::string v = "-0,01 RUB";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 6);
            assert(err == std::ios_base::goodbit);
            assert(ex == -1);
        }
        {   // negative one, showbase
            std::string v = "-0,01 RUB";
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
            std::noshowbase(ios);
        }
        {   // positive, showbase
            std::string v = "1 234 567,89 RUB";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 13);
            assert(err == std::ios_base::goodbit);
            assert(ex == 123456789);
        }
        {   // positive, showbase
            std::string v = "1 234 567,89 RUB";
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = "-1 234 567,89 RUB";
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = "-1 234 567,89 " + symbol;
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 14);
            assert(err == std::ios_base::failbit);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = "-1 234 567,89 " + symbol;
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 14);
            assert(err == std::ios_base::goodbit);
            assert(ex == -123456789);
        }
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    std::wstring wsymbol(LocaleHelpers::currency_symbol_ru_RU());
    {
        const my_facetw f(1);
        // wchar_t, national
        {   // zero
            std::wstring v = L"0,00";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
        }
        {   // negative one
            std::wstring v = L"-0,01 ";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
        }
        {   // positive
            std::wstring v = convert_thousands_sep(L"1 234 567,89 ");
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
        }
        {   // negative
            std::wstring v = convert_thousands_sep(L"-1 234 567,89 ");
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
        }
        {   // negative
            std::wstring v = L"-1234567,89 ";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
        }
        {   // zero, showbase
            std::wstring v = L"0,00 " + wsymbol;
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 5);
            assert(err == std::ios_base::goodbit);
            assert(ex == 0);
        }
        {   // zero, showbase
            std::wstring v = L"0,00 " + wsymbol;
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
            std::noshowbase(ios);
        }
        {   // negative one, showbase
            std::wstring v = L"-0,01 " + wsymbol;
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 6);
            assert(err == std::ios_base::goodbit);
            assert(ex == -1);
        }
        {   // negative one, showbase
            std::wstring v = L"-0,01 " + wsymbol;
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
            std::noshowbase(ios);
        }
        {   // positive, showbase
            std::wstring v = convert_thousands_sep(L"1 234 567,89 ") + wsymbol;
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 13);
            assert(err == std::ios_base::goodbit);
            assert(ex == 123456789);
        }
        {   // positive, showbase
            std::wstring v = convert_thousands_sep(L"1 234 567,89 ") + wsymbol;
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = convert_thousands_sep(L"-1 234 567,89 ") + wsymbol;
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = convert_thousands_sep(L"-1 234 567,89 RUB");
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 14);
            assert(err == std::ios_base::failbit);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = convert_thousands_sep(L"-1 234 567,89 RUB");
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 14);
            assert(err == std::ios_base::goodbit);
            assert(ex == -123456789);
        }
    }
    {
        const my_facetw f(1);
        // wchar_t, international
        {   // zero
            std::wstring v = L"0,00";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
        }
        {   // negative one
            std::wstring v = L"-0,01 ";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
        }
        {   // positive
            std::wstring v = convert_thousands_sep(L"1 234 567,89 ");
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
        }
        {   // negative
            std::wstring v = convert_thousands_sep(L"-1 234 567,89 ");
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
        }
        {   // negative
            std::wstring v = L"-1234567,89 ";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
        }
        {   // zero, showbase
            std::wstring v = L"0,00 RUB";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 5);
            assert(err == std::ios_base::goodbit);
            assert(ex == 0);
        }
        {   // zero, showbase
            std::wstring v = L"0,00 RUB";
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
            std::noshowbase(ios);
        }
        {   // negative one, showbase
            std::wstring v = L"-0,01 RUB";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 6);
            assert(err == std::ios_base::goodbit);
            assert(ex == -1);
        }
        {   // negative one, showbase
            std::wstring v = L"-0,01 RUB";
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
            std::noshowbase(ios);
        }
        {   // positive, showbase
            std::wstring v = convert_thousands_sep(L"1 234 567,89 RUB");
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 13);
            assert(err == std::ios_base::goodbit);
            assert(ex == 123456789);
        }
        {   // positive, showbase
            std::wstring v = convert_thousands_sep(L"1 234 567,89 RUB");
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = convert_thousands_sep(L"-1 234 567,89 RUB");
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = convert_thousands_sep(L"-1 234 567,89 ") + wsymbol;
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 14);
            assert(err == std::ios_base::failbit);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = convert_thousands_sep(L"-1 234 567,89 ") + wsymbol;
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 14);
            assert(err == std::ios_base::goodbit);
            assert(ex == -123456789);
        }
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
