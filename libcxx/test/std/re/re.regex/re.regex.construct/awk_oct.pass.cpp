//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// template <class ST, class SA>
//    basic_regex(const basic_string<charT, ST, SA>& s);

#include <regex>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
    using std::regex_constants::awk;

    assert(std::regex_match("\4", std::regex("\\4", awk)));
    assert(std::regex_match("\41", std::regex("\\41", awk)));
    assert(std::regex_match("\141", std::regex("\\141", awk)));
    assert(std::regex_match("\141" "1", std::regex("\\1411", awk)));

  return 0;
}
