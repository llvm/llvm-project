//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// class time_get_byname<charT, InputIterator>

// iter_type
// get_monthname(iter_type s, iter_type end, ios_base& str,
//               ios_base::iostate& err, tm* t) const;

#include <locale>
#include <cassert>
#include "iterators.h"

#include "../../../../platform_support.h" // locale name macros

typedef input_iterator<const char*> I;

typedef std::time_get_byname<char, I> F;

class my_facet
    : public F
{
public:
    explicit my_facet(const std::string& nm, std::size_t refs = 0)
        : F(nm, refs) {}
};

int main()
{
    std::ios ios(0);
    std::ios_base::iostate err;
    std::tm t;
    {
        const my_facet f(LOCALE_en_US_UTF_8, 1);
        const char in[] = "June";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_mon == 5);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_fr_FR_UTF_8, 1);
        const char in[] = "juin";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_mon == 5);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_ru_RU_UTF_8, 1);
        const char in[] = "\xD0\xB8\xD1\x8E\xD0\xBD\xD1\x8F";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_mon == 5);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_zh_CN_UTF_8, 1);
        const char in[] = "\xE5\x85\xAD\xE6\x9C\x88";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_monthname(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_mon == 5);
        assert(err == std::ios_base::eofbit);
    }
}
