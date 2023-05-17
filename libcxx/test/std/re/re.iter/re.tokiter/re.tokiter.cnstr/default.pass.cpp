//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class regex_token_iterator<BidirectionalIterator, charT, traits>

// regex_token_iterator();

#include <regex>
#include <cassert>
#include "test_macros.h"

template <class CharT>
void
test()
{
    typedef std::regex_token_iterator<const CharT*> I;
    I i1;
    assert(i1 == I());
}

int main(int, char**)
{
    test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<wchar_t>();
#endif

    return 0;
}
