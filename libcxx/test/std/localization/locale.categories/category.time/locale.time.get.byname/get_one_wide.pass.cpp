//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NetBSD does not support LC_TIME at the moment
// XFAIL: netbsd

// XFAIL: LIBCXX-FREEBSD-FIXME

// XFAIL: no-wide-characters

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ru_RU.UTF-8
// REQUIRES: locale.zh_CN.UTF-8

// <locale>

// class time_get_byname<charT, InputIterator>

// iter_type get(iter_type s, iter_type end, ios_base& f,
//               ios_base::iostate& err, tm *t, char format, char modifier = 0) const;

#include <locale>
#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

#include "platform_support.h" // locale name macros

typedef cpp17_input_iterator<const wchar_t*> I;

typedef std::time_get_byname<wchar_t, I> F;

class my_facet
    : public F
{
public:
    explicit my_facet(const std::string& nm, std::size_t refs = 0)
        : F(nm, refs) {}
};

int main(int, char**)
{
    std::ios ios(0);
    std::ios_base::iostate err;
    std::tm t;
    {
        const my_facet f(LOCALE_en_US_UTF_8, 1);
#ifdef _WIN32
        // On Windows, the "%c" format lacks the leading week day, which
        // means that t.tm_wday doesn't get set when parsing the string.
        const wchar_t in[] = L"12/31/2061 11:55:59 PM";
#elif defined(TEST_HAS_GLIBC)
        const wchar_t in[] = L"Sat 31 Dec 2061 11:55:59 PM";
#elif defined(_AIX)
        const wchar_t in[] = L"Dec 31, 2061 at 11:55:59 PM";
#else
        const wchar_t in[] = L"Sat Dec 31 23:55:59 2061";
#endif
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t, 'c');
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_sec == 59);
        assert(t.tm_min == 55);
        assert(t.tm_hour == 23);
        assert(t.tm_mday == 31);
        assert(t.tm_mon == 11);
        assert(t.tm_year == 161);
#if !defined(_WIN32) && !defined(_AIX)
        assert(t.tm_wday == 6);
#endif
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_en_US_UTF_8, 1);
#if defined(_WIN32) || defined(TEST_HAS_GLIBC) || defined(_AIX)
        const wchar_t in[] = L"11:55:59 PM";
#else
        const wchar_t in[] = L"23:55:59";
#endif
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t, 'X');
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_sec == 59);
        assert(t.tm_min == 55);
        assert(t.tm_hour == 23);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_fr_FR_UTF_8, 1);
#ifdef _WIN32
        const wchar_t in[] = L"31/12/2061 23:55:59";
#elif defined(TEST_HAS_GLIBC)
        const wchar_t in[] = L"sam. 31 d" L"\xE9" L"c. 2061 23:55:59";
#elif defined(_AIX)
        const wchar_t in[] = L"31 d" L"\xE9" L"c. 2061" L"\xE0" L"23:55:59";
#else
        const wchar_t in[] = L"Sam 31 d" L"\xE9" L"c 23:55:59 2061";
#endif
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t, 'c');
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_sec == 59);
        assert(t.tm_min == 55);
        assert(t.tm_hour == 23);
        assert(t.tm_mday == 31);
        assert(t.tm_mon == 11);
        assert(t.tm_year == 161);
#if !defined(_WIN32) && !defined(_AIX)
        assert(t.tm_wday == 6);
#endif
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_fr_FR_UTF_8, 1);
        const wchar_t in[] = L"23:55:59";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t, 'X');
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_sec == 59);
        assert(t.tm_min == 55);
        assert(t.tm_hour == 23);
        assert(err == std::ios_base::eofbit);
    }
#ifdef __APPLE__
    {
        const my_facet f("ru_RU", 1);
        const wchar_t in[] = L"\x441\x443\x431\x431\x43E\x442\x430"
                          L", 31 "
                          L"\x434\x435\x43A\x430\x431\x440\x44F"
                          L" 2061 "
                          L"\x433"
                          L". 23:55:59";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t, 'c');
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_sec == 59);
        assert(t.tm_min == 55);
        assert(t.tm_hour == 23);
        assert(t.tm_mday == 31);
        assert(t.tm_mon == 11);
        assert(t.tm_year == 161);
        assert(t.tm_wday == 6);
        assert(err == std::ios_base::eofbit);
    }
#endif
    {
        const my_facet f(LOCALE_ru_RU_UTF_8, 1);
        const wchar_t in[] = L"23:55:59";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t, 'X');
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_sec == 59);
        assert(t.tm_min == 55);
        assert(t.tm_hour == 23);
        assert(err == std::ios_base::eofbit);
    }
#ifdef __APPLE__
    {
        const my_facet f("zh_CN", 1);
        const wchar_t in[] = L"\x516D"
                          L" 12/31 23:55:59 2061";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t, 'c');
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_sec == 59);
        assert(t.tm_min == 55);
        assert(t.tm_hour == 23);
        assert(t.tm_mday == 31);
        assert(t.tm_mon == 11);
        assert(t.tm_year == 161);
        assert(t.tm_wday == 6);
        assert(err == std::ios_base::eofbit);
    }
#endif
    {
        const my_facet f(LOCALE_zh_CN_UTF_8, 1);
#ifdef _WIN32
        const wchar_t in[] = L"23:55:59";
#elif defined(_AIX)
        const wchar_t in[] = L"\x4E0B\x5348" L"11:55:59";
#else
        const wchar_t in[] = L"23" L"\x65F6" L"55" L"\x5206" L"59" L"\x79D2";
#endif
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t, 'X');
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_sec == 59);
        assert(t.tm_min == 55);
#if defined(_AIX)
        assert(t.tm_hour == 11);
#else
        assert(t.tm_hour == 23);
#endif
        assert(err == std::ios_base::eofbit);
    }

  return 0;
}
