//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class match_results<BidirectionalIterator, Allocator>

// match_results(const Allocator& a = Allocator());

#include <regex>
#include <cassert>
#include "test_macros.h"

template <class CharT>
void
test()
{
    std::match_results<const CharT*> m;
    assert(m.size() == 0);
    assert(m.str() == std::basic_string<CharT>());
    assert(m.get_allocator() == std::allocator<std::sub_match<const CharT*> >());
}

int main(int, char**)
{
    test<char>();
    test<wchar_t>();

  return 0;
}
