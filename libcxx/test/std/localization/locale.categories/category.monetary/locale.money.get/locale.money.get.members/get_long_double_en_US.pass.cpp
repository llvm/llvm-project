//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class money_get<charT, InputIterator>

// iter_type get(iter_type b, iter_type e, bool intl, ios_base& iob,
//               ios_base::iostate& err, long double& v) const;

// Bionic has minimal locale support, investigate this later.
// XFAIL: LIBCXX-ANDROID-FIXME

// REQUIRES: locale.en_US.UTF-8

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
            std::string v = "0.00";
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
            std::string v = LocaleHelpers::negate_en_US("0.01");
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
            std::string v = "1,234,567.89";
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
            std::string v = LocaleHelpers::negate_en_US("1,234,567.89");
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
            std::string v = LocaleHelpers::negate_en_US("1234567.89");
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
            std::string v = "$0.00";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
        }
        {   // zero, showbase
            std::string v = "$0.00";
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
            std::string v = LocaleHelpers::negate_en_US("$0.01");
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
        }
        {   // negative one, showbase
            std::string v = LocaleHelpers::negate_en_US("$0.01");
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
            std::string v = "$1,234,567.89";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
        }
        {   // positive, showbase
            std::string v = "$1,234,567.89";
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
            std::string v = LocaleHelpers::negate_en_US("$1,234,567.89");
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
            std::string v = LocaleHelpers::negate_en_US("USD 1,234,567.89");
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 1);
            assert(err == std::ios_base::failbit);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = LocaleHelpers::negate_en_US("USD 1,234,567.89");
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 1);
            assert(err == std::ios_base::failbit);
        }
    }
    {
        const my_facet f(1);
        // char, international
        {   // zero
            std::string v = "0.00";
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
            std::string v = LocaleHelpers::negate_en_US("0.01");
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
            std::string v = "1,234,567.89";
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
            std::string v = LocaleHelpers::negate_en_US("1,234,567.89");
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
            std::string v = LocaleHelpers::negate_en_US("1234567.89");
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
            std::string v = "USD 0.00";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
        }
        {   // zero, showbase
            std::string v = "USD 0.00";
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
            std::string v = LocaleHelpers::negate_en_US("USD 0.01");
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
        }
        {   // negative one, showbase
            std::string v = LocaleHelpers::negate_en_US("USD 0.01");
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
            std::string v = "USD 1,234,567.89";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
        }
        {   // positive, showbase
            std::string v = "USD 1,234,567.89";
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
            std::string v = LocaleHelpers::negate_en_US("USD 1,234,567.89");
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
            std::string v = LocaleHelpers::negate_en_US("$1,234,567.89");
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 1);
            assert(err == std::ios_base::failbit);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = LocaleHelpers::negate_en_US("$1,234,567.89");
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 1);
            assert(err == std::ios_base::failbit);
        }
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        const my_facetw f(1);
        // wchar_t, national
        {   // zero
            std::wstring v = L"0.00";
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
            std::wstring v = LocaleHelpers::negate_en_US(L"0.01");
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
            std::wstring v = L"1,234,567.89";
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
            std::wstring v = LocaleHelpers::negate_en_US(L"1,234,567.89");
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
            std::wstring v = LocaleHelpers::negate_en_US(L"1234567.89");
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
            std::wstring v = L"$0.00";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
        }
        {   // zero, showbase
            std::wstring v = L"$0.00";
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
            std::wstring v = LocaleHelpers::negate_en_US(L"$0.01");
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
        }
        {   // negative one, showbase
            std::wstring v = LocaleHelpers::negate_en_US(L"$0.01");
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
            std::wstring v = L"$1,234,567.89";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
        }
        {   // positive, showbase
            std::wstring v = L"$1,234,567.89";
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
            std::wstring v = LocaleHelpers::negate_en_US(L"$1,234,567.89");
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
            std::wstring v = LocaleHelpers::negate_en_US(L"USD 1,234,567.89");
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 1);
            assert(err == std::ios_base::failbit);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = LocaleHelpers::negate_en_US(L"USD 1,234,567.89");
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 1);
            assert(err == std::ios_base::failbit);
        }
    }
    {
        const my_facetw f(1);
        // wchar_t, international
        {   // zero
            std::wstring v = L"0.00";
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
            std::wstring v = LocaleHelpers::negate_en_US(L"0.01");
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
            std::wstring v = L"1,234,567.89";
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
            std::wstring v = LocaleHelpers::negate_en_US(L"1,234,567.89");
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
            std::wstring v = LocaleHelpers::negate_en_US(L"1234567.89");
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
            std::wstring v = L"USD 0.00";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
        }
        {   // zero, showbase
            std::wstring v = L"USD 0.00";
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
            std::wstring v = LocaleHelpers::negate_en_US(L"USD 0.01");
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
        }
        {   // negative one, showbase
            std::wstring v = LocaleHelpers::negate_en_US(L"USD 0.01");
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
            std::wstring v = L"USD 1,234,567.89";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
        }
        {   // positive, showbase
            std::wstring v = L"USD 1,234,567.89";
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
            std::wstring v = LocaleHelpers::negate_en_US(L"USD 1,234,567.89");
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
            std::wstring v = LocaleHelpers::negate_en_US(L"$1,234,567.89");
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 1);
            assert(err == std::ios_base::failbit);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = LocaleHelpers::negate_en_US(L"$1,234,567.89");
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 1);
            assert(err == std::ios_base::failbit);
        }
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS
    {
      // test for err
      const my_facet f(1);
      std::string v = "1.23";
      typedef cpp17_input_iterator<const char*> I;
      long double ex = -1L;
      std::ios_base::iostate err = std::ios_base::goodbit;
      f.get(I(v.data()), I(v.data()), false, ios, err, ex);
      assert(err == (std::ios_base::failbit | std::ios_base::eofbit));
      assert(ex == -1L);
    }

  return 0;
}
