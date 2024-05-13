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

// XFAIL: no-wide-characters

#include <locale>
#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

typedef cpp17_input_iterator<const wchar_t*> I;

typedef std::time_get<wchar_t, I> F;

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
        const wchar_t in[] = L"Jan";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 0);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"Feb";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 1);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"Mar";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 2);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"Apr";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 3);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"May";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 4);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"Jun";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 5);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"Jul";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 6);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"Aug";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 7);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"Sep";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 8);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"Oct";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 9);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"Nov";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 10);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"Dec";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 11);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"January";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+7);
        assert(t.tm_mon == 0);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"February";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+8);
        assert(t.tm_mon == 1);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"March";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+5);
        assert(t.tm_mon == 2);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"April";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+5);
        assert(t.tm_mon == 3);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"May";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+3);
        assert(t.tm_mon == 4);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"June";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+4);
        assert(t.tm_mon == 5);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"July";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+4);
        assert(t.tm_mon == 6);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"August";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+6);
        assert(t.tm_mon == 7);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"September";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+9);
        assert(t.tm_mon == 8);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"October";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+7);
        assert(t.tm_mon == 9);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"November";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+8);
        assert(t.tm_mon == 10);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"December";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+8);
        assert(t.tm_mon == 11);
        assert(err == std::ios_base::eofbit);
    }
    {
        const wchar_t in[] = L"Decemper";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(base(i) == in+5);
        assert(t.tm_mon == 0);
        assert(err == std::ios_base::failbit);
    }

  return 0;
}
