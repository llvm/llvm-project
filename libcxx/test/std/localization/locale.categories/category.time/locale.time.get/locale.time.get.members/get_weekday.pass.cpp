//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class time_get<charT, InputIterator>

// iter_type
// get_weekday(iter_type s, iter_type end, ios_base& str,
//             ios_base::iostate& err, tm* t) const;

#include <locale>
#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

typedef input_iterator<const char*> I;

typedef std::time_get<char, I> F;

class my_facet
    : public F
{
public:
    explicit my_facet(std::size_t refs = 0)
        : F(refs) {}
};

int main(int, char**)
{
    const my_facet f(1);
    std::ios ios(0);
    std::ios_base::iostate err;
    std::tm t;
    {
        const char in[] = "Sun";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_wday == 0);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Suny";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_wday == 0);
        assert(err == std::ios_base::goodbit);
    }
    {
        const char in[] = "Sund";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+4);
        assert(t.tm_wday == 0);
        assert(err == (std::ios_base::failbit | std::ios_base::eofbit));
    }
    {
        const char in[] = "sun";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_wday == 0);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "sunday";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+6);
        assert(t.tm_wday == 0);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Mon";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_wday == 1);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Mony";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_wday == 1);
        assert(err == std::ios_base::goodbit);
    }
    {
        const char in[] = "Mond";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+4);
        assert(t.tm_wday == 0);
        assert(err == (std::ios_base::failbit | std::ios_base::eofbit));
    }
    {
        const char in[] = "mon";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_wday == 1);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "monday";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+6);
        assert(t.tm_wday == 1);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Tue";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_wday == 2);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Tuesday";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+7);
        assert(t.tm_wday == 2);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Wed";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_wday == 3);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Wednesday";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+9);
        assert(t.tm_wday == 3);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Thu";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_wday == 4);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Thursday";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+8);
        assert(t.tm_wday == 4);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Fri";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_wday == 5);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Friday";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+6);
        assert(t.tm_wday == 5);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Sat";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_wday == 6);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Saturday";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_weekday(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+8);
        assert(t.tm_wday == 6);
        assert(err == std::ios_base::eofbit);
    }

  return 0;
}
