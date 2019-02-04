//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// <regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// template <class ST, class SA>
//    basic_regex(const basic_string<charT, ST, SA>& s);

#include <regex>
#include <cassert>
#include "test_macros.h"

static bool error_badbackref_thrown(const char *pat)
{
    bool result = false;
    try {
        std::regex re(pat);
    } catch (const std::regex_error &ex) {
        result = (ex.code() == std::regex_constants::error_backref);
    }
    return result;
}

int main()
{
    assert(error_badbackref_thrown("\\1abc"));      // no references
    assert(error_badbackref_thrown("ab(c)\\2def")); // only one reference
    assert(error_badbackref_thrown("\\800000000000000000000000000000")); // overflows

//  this should NOT throw, because we only should look at the '1'
//  See https://bugs.llvm.org/show_bug.cgi?id=31387
    {
    const char *pat1 = "a(b)c\\1234";
    std::regex re(pat1, pat1 + 7); // extra chars after the end.
    }
}
