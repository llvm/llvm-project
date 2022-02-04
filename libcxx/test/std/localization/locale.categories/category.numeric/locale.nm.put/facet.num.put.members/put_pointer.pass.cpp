//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class num_put<charT, OutputIterator>

// iter_type put(iter_type s, ios_base& iob, char_type fill, void* v) const;

#include <locale>
#include <ios>
#include <cassert>
#include <streambuf>
#include "test_macros.h"
#include "test_iterators.h"

typedef std::num_put<char, cpp17_output_iterator<char*> > F;

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
    {
        std::ios ios(0);
        void* v = 0;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        char expected_str[32] = {};
        // num_put::put uses %p for pointer types, but the exact format of %p is
        // implementation defined behavior for the C library. Compare output to
        // snprintf for portability.
        int rc = snprintf(expected_str, sizeof(expected_str), "%p", v);
        assert(rc > 0);
        assert(ex == expected_str);
    }

  return 0;
}
