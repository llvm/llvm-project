//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// <regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// template <class ST, class SA>
//    basic_regex(const basic_string<charT, ST, SA>& s);

#include <regex>
#include <cassert>
#include "test_macros.h"

static bool error_badbackref_thrown(const char *pat, std::regex::flag_type f)
{
    bool result = false;
    try {
        std::regex re(pat, f);
    } catch (const std::regex_error &ex) {
        result = (ex.code() == std::regex_constants::error_backref);
    }
    return result;
}

int main(int, char**)
{
//  no references
    assert(error_badbackref_thrown("\\1abc", std::regex_constants::ECMAScript));
    assert(error_badbackref_thrown("\\1abd", std::regex::basic));
    assert(error_badbackref_thrown("\\1abd", std::regex::extended));
    assert(error_badbackref_thrown("\\1abd", std::regex::awk) == false);
    assert(error_badbackref_thrown("\\1abd", std::regex::grep));
    assert(error_badbackref_thrown("\\1abd", std::regex::egrep));

//  only one reference
    assert(error_badbackref_thrown("ab(c)\\2def", std::regex_constants::ECMAScript));
    assert(error_badbackref_thrown("ab\\(c\\)\\2def", std::regex_constants::basic));
    assert(error_badbackref_thrown("ab(c)\\2def", std::regex_constants::extended));
    assert(error_badbackref_thrown("ab\\(c\\)\\2def", std::regex_constants::awk) == false);
    assert(error_badbackref_thrown("ab(c)\\2def", std::regex_constants::awk) == false);
    assert(error_badbackref_thrown("ab\\(c\\)\\2def", std::regex_constants::grep));
    assert(error_badbackref_thrown("ab(c)\\2def", std::regex_constants::egrep));


    assert(error_badbackref_thrown("\\800000000000000000000000000000", std::regex_constants::ECMAScript)); // overflows

//  this should NOT throw, because we only should look at the '1'
//  See https://llvm.org/PR31387
    {
    const char *pat1 = "a(b)c\\1234";
    std::regex re(pat1, pat1 + 7); // extra chars after the end.
    }

//  reference before group
    assert(error_badbackref_thrown("\\1(abc)", std::regex_constants::ECMAScript));
    assert(error_badbackref_thrown("\\1\\(abd\\)", std::regex::basic));
    assert(error_badbackref_thrown("\\1(abd)", std::regex::extended));
    assert(error_badbackref_thrown("\\1(abd)", std::regex::awk) == false);
    assert(error_badbackref_thrown("\\1\\(abd\\)", std::regex::awk) == false);
    assert(error_badbackref_thrown("\\1\\(abd\\)", std::regex::grep));
    assert(error_badbackref_thrown("\\1(abd)", std::regex::egrep));

//  reference limit
    assert(error_badbackref_thrown("(cat)\\10", std::regex::ECMAScript));
    assert(error_badbackref_thrown("\\(cat\\)\\10", std::regex::basic) == false);
    assert(error_badbackref_thrown("(cat)\\10", std::regex::extended) == false);
    assert(error_badbackref_thrown("\\(cat\\)\\10", std::regex::awk) == false);
    assert(error_badbackref_thrown("(cat)\\10", std::regex::awk) == false);
    assert(error_badbackref_thrown("\\(cat\\)\\10", std::regex::grep) == false);
    assert(error_badbackref_thrown("(cat)\\10", std::regex::egrep) == false);

//  https://llvm.org/PR34297
    assert(error_badbackref_thrown("(cat)\\1", std::regex::basic));
    assert(error_badbackref_thrown("\\(cat\\)\\1", std::regex::basic) == false);
    assert(error_badbackref_thrown("(cat)\\1", std::regex::extended) == false);
    assert(error_badbackref_thrown("\\(cat\\)\\1", std::regex::extended));
    assert(error_badbackref_thrown("(cat)\\1", std::regex::awk) == false);
    assert(error_badbackref_thrown("\\(cat\\)\\1", std::regex::awk) == false);
    assert(error_badbackref_thrown("(cat)\\1", std::regex::grep));
    assert(error_badbackref_thrown("\\(cat\\)\\1", std::regex::grep) == false);
    assert(error_badbackref_thrown("(cat)\\1", std::regex::egrep) == false);
    assert(error_badbackref_thrown("\\(cat\\)\\1", std::regex::egrep));

  return 0;
}
