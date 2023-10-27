//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iomanip>

// quoted

// UNSUPPORTED: c++03, c++11

#include <iomanip>
#include <sstream>
#include <string>

//  Test that mismatches in the traits between the quoted object and the dest string are diagnosed.

template <class charT>
struct test_traits {
    typedef charT char_type;
};

void round_trip ( const char *p ) {
    std::stringstream ss;
    ss << std::quoted(p);
    std::basic_string<char, test_traits<char>> s;
    ss >> std::quoted(s); // expected-error {{invalid operands to binary expression}}
}

void f() {
    round_trip("Hi Mom");
}
