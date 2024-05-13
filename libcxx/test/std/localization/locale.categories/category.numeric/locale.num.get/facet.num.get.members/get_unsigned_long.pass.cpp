//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class num_get<charT, InputIterator>

// iter_type get(iter_type in, iter_type end, ios_base&,
//               ios_base::iostate& err, unsigned long& v) const;

#include <locale>
#include <ios>
#include <cassert>
#include <streambuf>
#include "test_macros.h"
#include "test_iterators.h"

typedef std::num_get<char, cpp17_input_iterator<const char*> > F;

class my_facet
    : public F
{
public:
    explicit my_facet(std::size_t refs = 0)
        : F(refs) {}
};

class my_numpunct
    : public std::numpunct<char>
{
public:
    my_numpunct() : std::numpunct<char>() {}

protected:
    virtual char_type do_thousands_sep() const {return '_';}
    virtual std::string do_grouping() const {return std::string("\1\2\3");}
};

int main(int, char**)
{
    const my_facet f(1);
    std::ios ios(0);
    unsigned long v = static_cast<unsigned long>(-1);
    {
        const char str[] = "0";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 0);
    }
    {
        const char str[] = "1";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 1);
    }
    std::hex(ios);
    {
        const char str[] = "0xFFFFFFFF";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 0xFFFFFFFF);
    }

  return 0;
}
