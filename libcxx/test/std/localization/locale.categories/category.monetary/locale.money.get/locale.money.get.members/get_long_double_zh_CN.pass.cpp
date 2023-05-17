//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NetBSD does not support LC_MONETARY at the moment
// XFAIL: netbsd

// XFAIL: LIBCXX-AIX-FIXME
// XFAIL: LIBCXX-FREEBSD-FIXME

// REQUIRES: locale.zh_CN.UTF-8

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
#endif

int main(int, char**)
{
    std::ios ios(0);
    std::string loc_name(LOCALE_zh_CN_UTF_8);
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
#ifdef _WIN32
    std::string currency_name = "CNY";
#else
    std::string currency_name = "CNY ";
#endif
    std::string currency_symbol(LocaleHelpers::currency_symbol_zh_CN());
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
            std::string v = "-0.01";
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
            std::string v = "-1,234,567.89";
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
            std::string v = "-1234567.89";
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
            std::string v = currency_symbol + "0.00";
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
            std::string v = currency_symbol + "0.00";
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
            std::string v = currency_symbol + "-0.01";
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
            std::string v = currency_symbol + "-0.01";
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
            std::string v = currency_symbol + "1,234,567.89";
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
            std::string v = currency_symbol + "1,234,567.89";
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
            std::string v = currency_symbol + "-1,234,567.89";
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
            std::string v = "CNY -1,234,567.89";
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 0);
            assert(err == std::ios_base::failbit);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = "CNY -1,234,567.89";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 0);
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
            std::string v = "-0.01";
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
            std::string v = "-1,234,567.89";
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
            std::string v = "-1234567.89";
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
            std::string v = "CNY 0.00";
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
            std::string v = currency_name + "0.00";
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
#ifdef TEST_HAS_GLIBC
            std::string v = "-" + currency_name + "0.01";
#else
            std::string v = currency_name + "-0.01";
#endif
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
#ifdef TEST_HAS_GLIBC
            std::string v = "-" + currency_name + "0.01";
#else
            std::string v = currency_name + "-0.01";
#endif
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
            std::string v = currency_name + "1,234,567.89";
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
            std::string v = currency_name + "1,234,567.89";
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
#ifdef TEST_HAS_GLIBC
            std::string v = "-" + currency_name + "1,234,567.89";
#else
            std::string v = currency_name + "-1,234,567.89";
#endif
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
            std::string v = currency_symbol + "-1,234,567.89";
            std::showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 0);
            assert(err == std::ios_base::failbit);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = currency_symbol + "-1,234,567.89";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 0);
            assert(err == std::ios_base::failbit);
        }
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
#ifdef _WIN32
    std::wstring w_currency_name = L"CNY";
#else
    std::wstring w_currency_name = L"CNY ";
#endif
    std::wstring w_currency_symbol(LocaleHelpers::currency_symbol_zh_CN());
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
            std::wstring v = L"-0.01";
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
            std::wstring v = L"-1,234,567.89";
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
            std::wstring v = L"-1234567.89";
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
            std::wstring v = w_currency_symbol + L"0.00";
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
            std::wstring v = w_currency_symbol + L"0.00";
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
            std::wstring v = w_currency_symbol + L"-0.01";
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
            std::wstring v = w_currency_symbol + L"-0.01";
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
            std::wstring v = w_currency_symbol + L"1,234,567.89";
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
            std::wstring v = w_currency_symbol + L"1,234,567.89";
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
            std::wstring v = w_currency_symbol + L"-1,234,567.89";
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
            std::wstring v = L"CNY -1,234,567.89";
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 0);
            assert(err == std::ios_base::failbit);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = L"CNY -1,234,567.89";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(base(iter) == v.data() + 0);
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
            std::wstring v = L"-0.01";
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
            std::wstring v = L"-1,234,567.89";
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
            std::wstring v = L"-1234567.89";
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
            std::wstring v = L"CNY 0.00";
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
            std::wstring v = w_currency_name + L"0.00";
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
#ifdef TEST_HAS_GLIBC
            std::wstring v = L"-" + w_currency_name + L"0.01";
#else
            std::wstring v = w_currency_name + L"-0.01";
#endif
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
#ifdef TEST_HAS_GLIBC
            std::wstring v = L"-" + w_currency_name + L"0.01";
#else
            std::wstring v = w_currency_name + L"-0.01";
#endif
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
            std::wstring v = w_currency_name + L"1,234,567.89";
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
            std::wstring v = w_currency_name + L"1,234,567.89";
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
#ifdef TEST_HAS_GLIBC
            std::wstring v = L"-" + w_currency_name + L"1,234,567.89";
#else
            std::wstring v = w_currency_name + L"-1,234,567.89";
#endif
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
            std::wstring v = w_currency_symbol + L"-1,234,567.89";
            std::showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 0);
            assert(err == std::ios_base::failbit);
            std::noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = w_currency_symbol + L"-1,234,567.89";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(base(iter) == v.data() + 0);
            assert(err == std::ios_base::failbit);
        }
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
