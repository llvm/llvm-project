//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class match_results<BidirectionalIterator, Allocator>

// match_results(const match_results& m);

#include <regex>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"

template <class CharT, class Allocator>
void
test(const Allocator& a)
{
    typedef std::match_results<const CharT*, Allocator> SM;
    SM m0(a);
    SM m1(m0);

    assert(m1.size()          == m0.size());
    assert(m1.ready()         == m0.ready());
    assert(m1.get_allocator() == m0.get_allocator());
}

int main(int, char**)
{
    test<char>   (std::allocator<std::sub_match<const char *> >());
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<wchar_t>(std::allocator<std::sub_match<const wchar_t *> >());
#endif

    test<char>   (test_allocator<std::sub_match<const char*> >(3));
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<wchar_t>(test_allocator<std::sub_match<const wchar_t*> >(3));
#endif

  return 0;
}
