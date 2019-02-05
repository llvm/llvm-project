//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class match_results<BidirectionalIterator, Allocator>

// template <class BidirectionalIterator, class Allocator>
//    bool
//    operator==(const match_results<BidirectionalIterator, Allocator>& m1,
//               const match_results<BidirectionalIterator, Allocator>& m2);

// template <class BidirectionalIterator, class Allocator>
//    bool
//    operator!=(const match_results<BidirectionalIterator, Allocator>& m1,
//               const match_results<BidirectionalIterator, Allocator>& m2);

#include <regex>
#include <cassert>
#include "test_macros.h"

void
test()
{
    std::match_results<const char*> m1;
    const char s[] = "abcdefghijk";
    assert(std::regex_search(s, m1, std::regex("cd((e)fg)hi")));
    std::match_results<const char*> m2;

    assert(m1 == m1);
    assert(m1 != m2);

    m2 = m1;

    assert(m1 == m2);
}

int main()
{
    test();
}
