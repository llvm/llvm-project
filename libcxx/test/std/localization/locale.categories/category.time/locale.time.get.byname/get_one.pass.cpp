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
// XFAIL: LIBCXX-AIX-FIXME

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

typedef cpp17_input_iterator<const char*> I;

typedef std::time_get_byname<char, I> F;

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
        const char in[] = "12/31/2061 11:55:59 PM";
#elif defined(TEST_HAS_GLIBC)
        const char in[] = "Sat 31 Dec 2061 11:55:59 PM";
#else
        const char in[] = "Sat Dec 31 23:55:59 2061";
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
#ifndef _WIN32
        assert(t.tm_wday == 6);
#endif
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_en_US_UTF_8, 1);
#if defined(_WIN32) || defined(TEST_HAS_GLIBC)
        const char in[] = "11:55:59 PM";
#else
        const char in[] = "23:55:59";
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
        const char in[] = "31/12/2061 23:55:59";
#elif defined(TEST_HAS_GLIBC)
        const char in[] = "sam. 31 d""\xC3\xA9""c. 2061 23:55:59";
#else
        const char in[] = "Sam 31 d""\xC3\xA9""c 23:55:59 2061";
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
#ifndef _WIN32
        assert(t.tm_wday == 6);
#endif
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_fr_FR_UTF_8, 1);
        const char in[] = "23:55:59";
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
        const my_facet f(LOCALE_ru_RU_UTF_8, 1);
#ifdef TEST_HAS_GLIBC
        const char in[] = "\xD0\xA1\xD0\xB1 31 \xD0\xB4\xD0\xB5\xD0\xBA 2061 23:55:59";
#elif defined(_WIN32)
        const char in[] = "31.12.2061 23:55:59";
#else
        const char in[] = "\xD1\x81\xD1\x83\xD0\xB1\xD0\xB1"
                          "\xD0\xBE\xD1\x82\xD0\xB0"
                          ", 31 "
                          "\xD0\xB4\xD0\xB5\xD0\xBA\xD0\xB0"
                          "\xD0\xB1\xD1\x80\xD1\x8F"
                          " 2061 "
                          "\xD0\xB3"
                          ". 23:55:59";
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
#ifndef _WIN32
        assert(t.tm_wday == 6);
#endif
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_ru_RU_UTF_8, 1);
        const char in[] = "23:55:59";
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
        const my_facet f(LOCALE_zh_CN_UTF_8, 1);
#ifdef TEST_HAS_GLIBC
        const char in[] = "2061" "\xE5\xB9\xB4" "12" "\xE6\x9C\x88" "31"
                          "\xE6\x97\xA5" " "
                          "\xE6\x98\x9F\xE6\x9c\x9F\xE5\x85\xAD"
                          " 23" "\xE6\x97\xB6" "55" "\xE5\x88\x86" "59"
                          "\xE7\xA7\x92";
#elif defined(_WIN32)
        const char in[] = "2061/12/31 23:55:59";
#else
        const char in[] = "\xE5\x85\xAD 12/31 23:55:59 2061";
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
#ifndef _WIN32
        assert(t.tm_wday == 6);
#endif
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_zh_CN_UTF_8, 1);
#if defined(_WIN32)
        const char in[] = "23:55:59";
#else
        const char in[] = "23""\xE6\x97\xB6""55""\xE5\x88\x86""59""\xE7\xA7\x92";
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

  return 0;
}
