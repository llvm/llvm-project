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
//               ios_base::iostate& err, bool& v) const;

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

class p1
    : public std::numpunct<char>
{
public:
    p1() : std::numpunct<char>() {}

protected:
    virtual string_type do_truename() const {return "a";}
    virtual string_type do_falsename() const {return "abb";}
};

class p2
    : public std::numpunct<char>
{
public:
    p2() : std::numpunct<char>() {}

protected:
    virtual string_type do_truename() const {return "a";}
    virtual string_type do_falsename() const {return "ab";}
};

int main(int, char**)
{
    const my_facet f(1);
    std::ios ios(0);
    {
        const char str[] = "1";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, b);
        assert(iter.base() == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(b == true);
    }
    {
        const char str[] = "0";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, b);
        assert(iter.base() == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(b == false);
    }
    {
        const char str[] = "12";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, b);
        assert(iter.base() == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(b == true);
    }
    {
        const char str[] = "*12";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, b);
        assert(iter.base() == str+0);
        assert(err == ios.failbit);
        assert(b == false);
    }
    std::boolalpha(ios);
    {
        const char str[] = "1";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, b);
        assert(iter.base() == str+0);
        assert(err == ios.failbit);
        assert(b == false);
    }
    {
        const char str[] = "true";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, b);
        assert(iter.base() == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(b == true);
    }
    {
        const char str[] = "false";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, b);
        assert(iter.base() == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(b == false);
    }
    ios.imbue(std::locale(ios.getloc(), new p1));
    {
        const char str[] = "a";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+1),
                  ios, err, b);
        assert(iter.base() == str+1);
        assert(err == ios.eofbit);
        assert(b == true);
    }
    {
        const char str[] = "abc";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+3),
                  ios, err, b);
        assert(iter.base() == str+2);
        assert(err == ios.failbit);
        assert(b == false);
    }
    {
        const char str[] = "acc";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+3),
                  ios, err, b);
        assert(iter.base() == str+1);
        assert(err == ios.goodbit);
        assert(b == true);
    }
    ios.imbue(std::locale(ios.getloc(), new p2));
    {
        const char str[] = "a";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+1),
                  ios, err, b);
        assert(iter.base() == str+1);
        assert(err == ios.eofbit);
        assert(b == true);
    }
    {
        const char str[] = "ab";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+2),
                  ios, err, b);
        assert(iter.base() == str+2);
        assert(err == ios.eofbit);
        assert(b == false);
    }
    {
        const char str[] = "abc";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+3),
                  ios, err, b);
        assert(iter.base() == str+2);
        assert(err == ios.goodbit);
        assert(b == false);
    }
    {
        const char str[] = "ac";
        std::ios_base::iostate err = ios.goodbit;
        bool b;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+2),
                  ios, err, b);
        assert(iter.base() == str+1);
        assert(err == ios.goodbit);
        assert(b == true);
    }

  return 0;
}
