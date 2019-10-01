//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// basic_regex(const charT* p, size_t len, flag_type f);

#include <regex>
#include <cassert>
#include "test_macros.h"

template <class CharT>
void
test(const CharT* p, std::size_t len, std::regex_constants::syntax_option_type f,
     unsigned mc)
{
    std::basic_regex<CharT> r(p, len, f);
    assert(r.flags() == f);
    assert(r.mark_count() == mc);
}

int main(int, char**)
{
    test("\\(a\\)", 5, std::regex_constants::basic, 1);
    test("\\(a[bc]\\)", 9, std::regex_constants::basic, 1);
    test("\\(a\\([bc]\\)\\)", 13, std::regex_constants::basic, 2);
    test("(a([bc]))", 9, std::regex_constants::basic, 0);

    test("\\(a\\)", 5, std::regex_constants::extended, 0);
    test("\\(a[bc]\\)", 9, std::regex_constants::extended, 0);
    test("\\(a\\([bc]\\)\\)", 13, std::regex_constants::extended, 0);
    test("(a([bc]))", 9, std::regex_constants::extended, 2);

    test("\\(a\\)", 5, std::regex_constants::ECMAScript, 0);
    test("\\(a[bc]\\)", 9, std::regex_constants::ECMAScript, 0);
    test("\\(a\\([bc]\\)\\)", 13, std::regex_constants::ECMAScript, 0);
    test("(a([bc]))", 9, std::regex_constants::ECMAScript, 2);

    test("\\(a\\)", 5, std::regex_constants::awk, 0);
    test("\\(a[bc]\\)", 9, std::regex_constants::awk, 0);
    test("\\(a\\([bc]\\)\\)", 13, std::regex_constants::awk, 0);
    test("(a([bc]))", 9, std::regex_constants::awk, 2);

    test("\\(a\\)", 5, std::regex_constants::grep, 1);
    test("\\(a[bc]\\)", 9, std::regex_constants::grep, 1);
    test("\\(a\\([bc]\\)\\)", 13, std::regex_constants::grep, 2);
    test("(a([bc]))", 9, std::regex_constants::grep, 0);

    test("\\(a\\)", 5, std::regex_constants::egrep, 0);
    test("\\(a[bc]\\)", 9, std::regex_constants::egrep, 0);
    test("\\(a\\([bc]\\)\\)", 13, std::regex_constants::egrep, 0);
    test("(a([bc]))", 9, std::regex_constants::egrep, 2);

  return 0;
}
