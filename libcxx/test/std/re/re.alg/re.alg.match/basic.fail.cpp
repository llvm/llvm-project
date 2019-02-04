//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

//   template <class ST, class SA, class Allocator, class charT, class traits>
//   bool regex_match(const basic_string<charT, ST, SA>&&,
//                    match_results<
//                      typename basic_string<charT, ST, SA>::const_iterator,
//                      Allocator>&,
//                    const basic_regex<charT, traits>&,
//                    regex_constants::match_flag_type =
//                      regex_constants::match_default) = delete;

#include <regex>
#include <cassert>
#include "test_macros.h"

#if TEST_STD_VER < 14
#error
#endif

int main()
{
    {
        std::smatch m;
        std::regex re{"*"};
        std::regex_match(std::string("abcde"), m, re);
    }
}
