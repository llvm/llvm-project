//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>
// UNSUPPORTED: c++03, c++11, c++14

// template<class ForwardIterator>
// basic_regex(ForwardIterator, ForwardIterator,
//             regex_constants::syntax_option_type = regex_constants::ECMAScript)
// -> basic_regex<typename iterator_traits<ForwardIterator>::value_type>;

#include <regex>
#include <string>
#include <iterator>
#include <cassert>
#include <cstddef>


int main(int, char**)
{
    // Test the explicit deduction guides
    {
    // basic_regex(ForwardIterator, ForwardIterator)
    // <int> is not an iterator
    std::basic_regex re(23, 34);   // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'basic_regex'}}
    }

    {
    // basic_regex(ForwardIterator, ForwardIterator, flag_type)
    // <double> is not an iterator
    std::basic_regex re(23.0, 34.0, std::regex_constants::basic);   // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'basic_regex'}}
    }

    return 0;
}
