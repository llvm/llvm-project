//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// basic_regex();

#include <regex>
#include <cassert>
#include "test_macros.h"

template <class CharT>
void
test()
{
    std::basic_regex<CharT> r;
    assert(r.flags() == std::regex_constants::ECMAScript);
    assert(r.mark_count() == 0);
}

int main(int, char**)
{
    test<char>();
    test<wchar_t>();

  return 0;
}
