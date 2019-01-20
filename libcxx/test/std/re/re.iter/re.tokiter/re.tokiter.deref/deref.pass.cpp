//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class regex_token_iterator<BidirectionalIterator, charT, traits>

// const value_type& operator*() const;

#include <regex>
#include <cassert>
#include "test_macros.h"

int main()
{
    {
        std::regex phone_numbers("\\d{3}-\\d{4}");
        const char phone_book[] = "start 555-1234, 555-2345, 555-3456 end";
        std::cregex_token_iterator i(std::begin(phone_book), std::end(phone_book)-1,
                                     phone_numbers, -1);
        assert(i != std::cregex_token_iterator());
        assert((*i).str() == "start ");
        ++i;
        assert(i != std::cregex_token_iterator());
        assert((*i).str() == ", ");
        ++i;
        assert(i != std::cregex_token_iterator());
        assert((*i).str() == ", ");
        ++i;
        assert(i != std::cregex_token_iterator());
        assert((*i).str() == " end");
        ++i;
        assert(i == std::cregex_token_iterator());
    }
    {
        std::regex phone_numbers("\\d{3}-\\d{4}");
        const char phone_book[] = "start 555-1234, 555-2345, 555-3456 end";
        std::cregex_token_iterator i(std::begin(phone_book), std::end(phone_book)-1,
                                     phone_numbers);
        assert(i != std::cregex_token_iterator());
        assert((*i).str() == "555-1234");
        ++i;
        assert(i != std::cregex_token_iterator());
        assert((*i).str() == "555-2345");
        ++i;
        assert(i != std::cregex_token_iterator());
        assert((*i).str() == "555-3456");
        ++i;
        assert(i == std::cregex_token_iterator());
    }
    {
        std::regex phone_numbers("\\d{3}-(\\d{4})");
        const char phone_book[] = "start 555-1234, 555-2345, 555-3456 end";
        std::cregex_token_iterator i(std::begin(phone_book), std::end(phone_book)-1,
                                     phone_numbers, 1);
        assert(i != std::cregex_token_iterator());
        assert((*i).str() == "1234");
        ++i;
        assert(i != std::cregex_token_iterator());
        assert((*i).str() == "2345");
        ++i;
        assert(i != std::cregex_token_iterator());
        assert((*i).str() == "3456");
        ++i;
        assert(i == std::cregex_token_iterator());
    }
}
