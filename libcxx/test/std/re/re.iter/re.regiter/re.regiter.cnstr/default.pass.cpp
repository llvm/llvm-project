//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class regex_iterator<BidirectionalIterator, charT, traits>

// regex_iterator();

#include <regex>
#include <cassert>
#include "test_macros.h"

template <class CharT>
void
test()
{
    typedef std::regex_iterator<const CharT*> I;
    I i1;
    assert(i1 == I());
}

int main(int, char**)
{
    test<char>();
    test<wchar_t>();

  return 0;
}
