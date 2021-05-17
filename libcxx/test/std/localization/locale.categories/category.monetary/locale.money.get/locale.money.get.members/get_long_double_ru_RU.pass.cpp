//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NetBSD does not support LC_MONETARY at the moment
// XFAIL: netbsd

// Failure related to GLIBC's use of U00A0 as mon_thousands_sep
// and U002E as mon_decimal_point.
// TODO: U00A0 should be investigated.
// Possibly related to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=16006
// XFAIL: linux

// XFAIL: LIBCXX-WINDOWS-FIXME

// REQUIRES: locale.ru_RU.UTF-8

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

#include "platform_support.h" // locale name macros

// TODO:
// Some of the assertions in this test are failing on Apple platforms.
// Until we figure out the problem and fix it, disable these tests on
// Apple platforms. Note that we're not using XFAIL or UNSUPPORTED markup
// here, because this test would otherwise be disabled on all platforms
// we test. To avoid this test becoming entirely stale, we just disable
// the parts that fail.
//
// See https://llvm.org/PR45739 for the bug tracking this.
#if defined(__APPLE__)
#   define APPLE_FIXME
#endif

typedef std::money_get<char, cpp17_input_iterator<const char*> > Fn;

class my_facet
    : public Fn
{
public:
    explicit my_facet(std::size_t refs = 0)
        : Fn(refs) {}
};

typedef std::money_get<wchar_t, cpp17_input_iterator<const wchar_t*> > Fw;

class my_facetw
    : public Fw
{
public:
    explicit my_facetw(std::size_t refs = 0)
        : Fw(refs) {}
};

int main(int, char**)
{
    std::ios ios(0);
    std::string loc_name(LOCALE_ru_RU_UTF_8);
    ios.imbue(std::locale(ios.getloc(),
                          new std::moneypunct_byname<char, false>(loc_name)));
    ios.imbue(std::locale(ios.getloc(),
                          new std::moneypunct_byname<char, true>(loc_name)));
    ios.imbue(std::locale(ios.getloc(),
                          new std::moneypunct_byname<wchar_t, false>(loc_name)));
    ios.imbue(std::locale(ios.getloc(),
                          new std::moneypunct_byname<wchar_t, true>(loc_name)));
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
            assert(iter.base() == v.data() + v.size());
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
            assert(iter.base() == v.data() + v.size());
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
            assert(iter.base() == v.data() + v.size());
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
            assert(iter.base() == v.data() + v.size());
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
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
        }
        {   // zero, showbase
            std::string v = "0,00 \xD1\x80\xD1\x83\xD0\xB1"".";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + 5);
            assert(err == std::ios_base::goodbit);
            assert(ex == 0);
        }
        {   // zero, showbase
            std::string v = "0,00 \xD1\x80\xD1\x83\xD0\xB1"".";
            showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
            noshowbase(ios);
        }
        {   // negative one, showbase
            std::string v = "-0,01 \xD1\x80\xD1\x83\xD0\xB1"".";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + 6);
            assert(err == std::ios_base::goodbit);
            assert(ex == -1);
        }
        {   // negative one, showbase
            std::string v = "-0,01 \xD1\x80\xD1\x83\xD0\xB1"".";
            showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
            noshowbase(ios);
        }
        {   // positive, showbase
            std::string v = "1 234 567,89 \xD1\x80\xD1\x83\xD0\xB1"".";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + 13);
            assert(err == std::ios_base::goodbit);
            assert(ex == 123456789);
        }
        {   // positive, showbase
            std::string v = "1 234 567,89 \xD1\x80\xD1\x83\xD0\xB1"".";
            showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
            noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = "-1 234 567,89 \xD1\x80\xD1\x83\xD0\xB1"".";
            showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
            noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = "-1 234 567,89 RUB ";
            showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + 14);
            assert(err == std::ios_base::failbit);
            noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = "-1 234 567,89 RUB ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + 14);
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
            assert(iter.base() == v.data() + v.size());
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
            assert(iter.base() == v.data() + v.size());
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
            assert(iter.base() == v.data() + v.size());
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
            assert(iter.base() == v.data() + v.size());
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
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
        }
        {   // zero, showbase
            std::string v = "0,00 RUB ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + 5);
            assert(err == std::ios_base::goodbit);
            assert(ex == 0);
        }
#if !defined(APPLE_FIXME)
        {   // zero, showbase
            std::string v = "0,00 RUB ";
            showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
            noshowbase(ios);
        }
#endif
        {   // negative one, showbase
            std::string v = "-0,01 RUB ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + 6);
            assert(err == std::ios_base::goodbit);
            assert(ex == -1);
        }
#if !defined(APPLE_FIXME)
        {   // negative one, showbase
            std::string v = "-0,01 RUB ";
            showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
            noshowbase(ios);
        }
#endif
        {   // positive, showbase
            std::string v = "1 234 567,89 RUB ";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + 13);
            assert(err == std::ios_base::goodbit);
            assert(ex == 123456789);
        }
#if !defined(APPLE_FIXME)
        {   // positive, showbase
            std::string v = "1 234 567,89 RUB ";
            showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
            noshowbase(ios);
        }
#endif
#if !defined(APPLE_FIXME)
        {   // negative, showbase
            std::string v = "-1 234 567,89 RUB ";
            showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
            noshowbase(ios);
        }
#endif
        {   // negative, showbase
            std::string v = "-1 234 567,89 \xD1\x80\xD1\x83\xD0\xB1"".";
            showbase(ios);
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + 14);
            assert(err == std::ios_base::failbit);
            noshowbase(ios);
        }
        {   // negative, showbase
            std::string v = "-1 234 567,89 \xD1\x80\xD1\x83\xD0\xB1"".";
            typedef cpp17_input_iterator<const char*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + 14);
            assert(err == std::ios_base::goodbit);
            assert(ex == -123456789);
        }
    }
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
            assert(iter.base() == v.data() + v.size());
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
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
        }
        {   // positive
            std::wstring v = L"1 234 567,89 ";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
        }
        {   // negative
            std::wstring v = L"-1 234 567,89 ";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
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
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
        }
        {   // zero, showbase
            std::wstring v = L"0,00 \x440\x443\x431"".";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + 5);
            assert(err == std::ios_base::goodbit);
            assert(ex == 0);
        }
        {   // zero, showbase
            std::wstring v = L"0,00 \x440\x443\x431"".";
            showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
            noshowbase(ios);
        }
        {   // negative one, showbase
            std::wstring v = L"-0,01 \x440\x443\x431"".";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + 6);
            assert(err == std::ios_base::goodbit);
            assert(ex == -1);
        }
        {   // negative one, showbase
            std::wstring v = L"-0,01 \x440\x443\x431"".";
            showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
            noshowbase(ios);
        }
        {   // positive, showbase
            std::wstring v = L"1 234 567,89 \x440\x443\x431"".";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + 13);
            assert(err == std::ios_base::goodbit);
            assert(ex == 123456789);
        }
        {   // positive, showbase
            std::wstring v = L"1 234 567,89 \x440\x443\x431"".";
            showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
            noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = L"-1 234 567,89 \x440\x443\x431"".";
            showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
            noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = L"-1 234 567,89 RUB ";
            showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + 14);
            assert(err == std::ios_base::failbit);
            noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = L"-1 234 567,89 RUB ";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                false, ios, err, ex);
            assert(iter.base() == v.data() + 14);
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
            assert(iter.base() == v.data() + v.size());
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
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
        }
        {   // positive
            std::wstring v = L"1 234 567,89 ";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
        }
        {   // negative
            std::wstring v = L"-1 234 567,89 ";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
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
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
        }
        {   // zero, showbase
            std::wstring v = L"0,00 RUB ";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + 5);
            assert(err == std::ios_base::goodbit);
            assert(ex == 0);
        }
#if !defined(APPLE_FIXME)
        {   // zero, showbase
            std::wstring v = L"0,00 RUB ";
            showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 0);
            noshowbase(ios);
        }
#endif
        {   // negative one, showbase
            std::wstring v = L"-0,01 RUB ";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + 6);
            assert(err == std::ios_base::goodbit);
            assert(ex == -1);
        }
#if !defined(APPLE_FIXME)
        {   // negative one, showbase
            std::wstring v = L"-0,01 RUB ";
            showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -1);
            noshowbase(ios);
        }
#endif
        {   // positive, showbase
            std::wstring v = L"1 234 567,89 RUB ";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + 13);
            assert(err == std::ios_base::goodbit);
            assert(ex == 123456789);
        }
#if !defined(APPLE_FIXME)
        {   // positive, showbase
            std::wstring v = L"1 234 567,89 RUB ";
            showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == 123456789);
            noshowbase(ios);
        }
#endif
#if !defined(APPLE_FIXME)
        {   // negative, showbase
            std::wstring v = L"-1 234 567,89 RUB ";
            showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + v.size());
            assert(err == std::ios_base::eofbit);
            assert(ex == -123456789);
            noshowbase(ios);
        }
#endif
        {   // negative, showbase
            std::wstring v = L"-1 234 567,89 \x440\x443\x431"".";
            showbase(ios);
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + 14);
            assert(err == std::ios_base::failbit);
            noshowbase(ios);
        }
        {   // negative, showbase
            std::wstring v = L"-1 234 567,89 \x440\x443\x431"".";
            typedef cpp17_input_iterator<const wchar_t*> I;
            long double ex;
            std::ios_base::iostate err = std::ios_base::goodbit;
            I iter = f.get(I(v.data()), I(v.data() + v.size()),
                                                true, ios, err, ex);
            assert(iter.base() == v.data() + 14);
            assert(err == std::ios_base::goodbit);
            assert(ex == -123456789);
        }
    }

  return 0;
}
