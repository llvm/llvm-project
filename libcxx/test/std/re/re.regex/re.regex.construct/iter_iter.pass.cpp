//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// template <class ForwardIterator>
//    basic_regex(ForwardIterator first, ForwardIterator last);

#include <regex>
#include <cassert>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
void
test(Iter first, Iter last, unsigned mc)
{
    std::basic_regex<typename std::iterator_traits<Iter>::value_type> r(first, last);
    assert(r.flags() == std::regex_constants::ECMAScript);
    assert(r.mark_count() == mc);
}

int main(int, char**)
{
    typedef forward_iterator<std::string::const_iterator> F;
    std::string s1("\\(a\\)");
    std::string s2("\\(a[bc]\\)");
    std::string s3("\\(a\\([bc]\\)\\)");
    std::string s4("(a([bc]))");

    test(F(s1.begin()), F(s1.end()), 0);
    test(F(s2.begin()), F(s2.end()), 0);
    test(F(s3.begin()), F(s3.end()), 0);
    test(F(s4.begin()), F(s4.end()), 2);

  return 0;
}
