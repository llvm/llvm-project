//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class regex_iterator<BidirectionalIterator, charT, traits>

// const value_type& operator*() const;

#include <regex>
#include <cassert>
#include "test_macros.h"

int main()
{
    {
        std::regex phone_numbers("\\d{3}-\\d{4}");
        const char phone_book[] = "555-1234, 555-2345, 555-3456";
        std::cregex_iterator i(std::begin(phone_book), std::end(phone_book), phone_numbers);
        assert(i != std::cregex_iterator());
        assert((*i).size() == 1);
        assert((*i).position() == 0);
        assert((*i).str() == "555-1234");
        ++i;
        assert(i != std::cregex_iterator());
        assert((*i).size() == 1);
        assert((*i).position() == 10);
        assert((*i).str() == "555-2345");
        ++i;
        assert(i != std::cregex_iterator());
        assert((*i).size() == 1);
        assert((*i).position() == 20);
        assert((*i).str() == "555-3456");
        ++i;
        assert(i == std::cregex_iterator());
    }
}
