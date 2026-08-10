//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// basic_regex& assign(const charT* ptr, size_t len, flag_type f);

#include <regex>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
    std::regex r0;
    r0.assign("(a([bc]))", 9);
    assert(r0.flags() == std::regex::ECMAScript);
    assert(r0.mark_count() == 2);

    std::regex r1;
    r1.assign("(a([bc]))", 9, std::regex::ECMAScript);
    assert(r1.flags() == std::regex::ECMAScript);
    assert(r1.mark_count() == 2);

    std::regex r2;
    r2.assign("(a([bc]))", 9, std::regex::extended);
    assert(r2.flags() == std::regex::extended);
    assert(r2.mark_count() == 2);

  return 0;
}
