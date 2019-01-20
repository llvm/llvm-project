//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class regex_iterator<BidirectionalIterator, charT, traits>

// regex_token_iterator(BidirectionalIterator a, BidirectionalIterator b,
//                      const regex_type&& re,
//                      int submatch = 0,
//                      regex_constants::match_flag_type m =
//                        regex_constants::match_default) = delete;

#include <regex>
#include <cassert>
#include "test_macros.h"

#if TEST_STD_VER < 14
#error
#endif

int main()
{
    {
        const char phone_book[] = "555-1234, 555-2345, 555-3456";
        std::cregex_iterator i(
            std::begin(phone_book), std::end(phone_book),
            std::regex("\\d{3}-\\d{4}"));
    }
}
