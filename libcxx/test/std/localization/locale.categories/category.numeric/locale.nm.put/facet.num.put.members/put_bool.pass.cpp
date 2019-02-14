//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class num_put<charT, OutputIterator>

// iter_type put(iter_type s, ios_base& iob, char_type fill, bool v) const;

#include <locale>
#include <ios>
#include <cassert>
#include <streambuf>
#include "test_iterators.h"

typedef std::num_put<char, output_iterator<char*> > F;

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
    virtual string_type do_truename() const {return "yes";}
    virtual string_type do_falsename() const {return "no";}
};

int main(int, char**)
{
    const my_facet f(1);
    {
        std::ios ios(0);
        {
            bool v = false;
            char str[50];
            output_iterator<char*> iter = f.put(output_iterator<char*>(str), ios, '*', v);
            std::string ex(str, iter.base());
            assert(ex == "0");
        }
        {
            bool v = true;
            char str[50];
            output_iterator<char*> iter = f.put(output_iterator<char*>(str), ios, '*', v);
            std::string ex(str, iter.base());
            assert(ex == "1");
        }
    }
    {
        std::ios ios(0);
        boolalpha(ios);
        {
            bool v = false;
            char str[50];
            output_iterator<char*> iter = f.put(output_iterator<char*>(str), ios, '*', v);
            std::string ex(str, iter.base());
            assert(ex == "false");
        }
        {
            bool v = true;
            char str[50];
            output_iterator<char*> iter = f.put(output_iterator<char*>(str), ios, '*', v);
            std::string ex(str, iter.base());
            assert(ex == "true");
        }
    }
    {
        std::ios ios(0);
        boolalpha(ios);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        {
            bool v = false;
            char str[50];
            output_iterator<char*> iter = f.put(output_iterator<char*>(str), ios, '*', v);
            std::string ex(str, iter.base());
            assert(ex == "no");
        }
        {
            bool v = true;
            char str[50];
            output_iterator<char*> iter = f.put(output_iterator<char*>(str), ios, '*', v);
            std::string ex(str, iter.base());
            assert(ex == "yes");
        }
    }

  return 0;
}
