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
// get_monthname(iter_type s, iter_type end, ios_base& str,
//               ios_base::iostate& err, tm* t) const;

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
        const char in[] = "Jan";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 0);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Feb";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 1);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Mar";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 2);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Apr";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 3);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "May";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 4);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Jun";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 5);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Jul";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 6);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Aug";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 7);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Sep";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 8);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Oct";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 9);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Nov";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 10);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Dec";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 11);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "January";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+7);
        assert(t.tm_mon == 0);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "February";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+8);
        assert(t.tm_mon == 1);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "March";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+5);
        assert(t.tm_mon == 2);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "April";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+5);
        assert(t.tm_mon == 3);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "May";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+3);
        assert(t.tm_mon == 4);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "June";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+4);
        assert(t.tm_mon == 5);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "July";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+4);
        assert(t.tm_mon == 6);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "August";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+6);
        assert(t.tm_mon == 7);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "September";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+9);
        assert(t.tm_mon == 8);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "October";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+7);
        assert(t.tm_mon == 9);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "November";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+8);
        assert(t.tm_mon == 10);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "December";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+8);
        assert(t.tm_mon == 11);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "Decemper";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+5);
        assert(t.tm_mon == 0);
        assert(err == std::ios_base::failbit);
    }

  return 0;
}
