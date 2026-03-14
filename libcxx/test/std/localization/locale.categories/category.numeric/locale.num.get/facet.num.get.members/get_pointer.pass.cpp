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
//               ios_base::iostate& err, void*& v) const;

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

int main(int, char**)
{
    const my_facet f(1);
    std::ios ios(0);
    {
        const char str[] = "0x0";
        std::ios_base::iostate err = ios.goodbit;
        void* p;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, p);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(p == 0);
    }
    {
        const char str[] = "0x73";
        std::ios_base::iostate err = ios.goodbit;
        void* p;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, p);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(p == (void*)0x73);
    }

  return 0;
}
