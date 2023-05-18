//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// basic_regex& assign(const basic_regex& that);

#include <regex>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
    std::regex r1("(a([bc]))");
    std::regex r2;
    r2.assign(r1);
    assert(r2.flags() == std::regex::ECMAScript);
    assert(r2.mark_count() == 2);
    assert(std::regex_search("ab", r2));

#ifndef TEST_HAS_NO_EXCEPTIONS
    bool caught = false;
    try { r2.assign("(def", std::regex::extended); }
    catch(std::regex_error &) { caught = true; }
    assert(caught);
    assert(r2.flags() == std::regex::ECMAScript);
    assert(r2.mark_count() == 2);
    assert(std::regex_search("ab", r2));
#endif

  return 0;
}
