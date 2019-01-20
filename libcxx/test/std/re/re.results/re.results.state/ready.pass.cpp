//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class match_results<BidirectionalIterator, Allocator>

// bool ready() const;

#include <regex>
#include <cassert>
#include "test_macros.h"

void
test1()
{
    std::match_results<const char*> m;
    const char s[] = "abcdefghijk";
    assert(m.ready() == false);
    std::regex_search(s, m, std::regex("cd((e)fg)hi"));
    assert(m.ready() == true);
}

void
test2()
{
    std::match_results<const char*> m;
    const char s[] = "abcdefghijk";
    assert(m.ready() == false);
    std::regex_search(s, m, std::regex("z"));
    assert(m.ready() == true);
}

int main()
{
    test1();
    test2();
}
