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

// iter_type
// get_date(iter_type s, iter_type end, ios_base& str,
//          ios_base::iostate& err, tm* t) const;

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
        const char in[] = "06/10/2009";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_date(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_mon == 5);
        assert(t.tm_mday == 10);
        assert(t.tm_year == 109);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_fr_FR_UTF_8, 1);
#if defined(_WIN32) || defined(TEST_HAS_GLIBC)
        const char in[] = "10/06/2009";
#else
        const char in[] = "10.06.2009";
#endif
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_date(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_mon == 5);
        assert(t.tm_mday == 10);
        assert(t.tm_year == 109);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_ru_RU_UTF_8, 1);
        const char in[] = "10.06.2009";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_date(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_mon == 5);
        assert(t.tm_mday == 10);
        assert(t.tm_year == 109);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_zh_CN_UTF_8, 1);
#ifdef TEST_HAS_GLIBC
        // There's no separator between month and day.
        const char in[] = "2009\u5e740610";
#else
        const char in[] = "2009/06/10";
#endif
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_date(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_mon == 5);
        assert(t.tm_mday == 10);
        assert(t.tm_year == 109);
        assert(err == std::ios_base::eofbit);
    }
    // Months must be > 0 and <= 12.
    {
        const my_facet f(LOCALE_en_US_UTF_8, 1);
        const char in[] = "00/21/2022";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_date(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
#if _LIBCPP_VERSION
          // libc++ points to the '/' after the month.
          assert(base(i) == in+2);
#else
          // libstdc++ points to the second character.
          assert(base(i) == in+1);
#endif
        // tm is not modified.
        assert(t.tm_mon == 0);
        assert(t.tm_mday == 0);
        assert(t.tm_year == 0);
        assert(err == std::ios_base::failbit);
    }
    {
        const my_facet f(LOCALE_en_US_UTF_8, 1);
        const char in[] = "13/21/2022";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_date(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
#if _LIBCPP_VERSION
          // libc++ points to the '/' after the month.
          assert(base(i) == in+2);
#else
          // libstdc++ points to the second character.
          assert(base(i) == in+1);
#endif
        assert(base(i) == in+2);
        assert(t.tm_mon == 0);
        assert(t.tm_mday == 0);
        assert(t.tm_year == 0);
        assert(err == std::ios_base::failbit);
    }
    // Leading zero is allowed.
    {
        const my_facet f(LOCALE_en_US_UTF_8, 1);
        const char in[] = "03/21/2022";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_date(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_mon == 2);
        assert(t.tm_mday == 21);
        assert(t.tm_year == 122);
        assert(err == std::ios_base::eofbit);
    }
  return 0;
}
