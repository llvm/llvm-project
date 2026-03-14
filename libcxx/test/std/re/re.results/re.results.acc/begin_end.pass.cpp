//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class match_results<BidirectionalIterator, Allocator>

// const_iterator begin() const;
// const_iterator end() const;

#include <regex>
#include <cassert>
#include <cstddef>
#include "test_macros.h"

void
test()
{
    std::match_results<const char*> m;
    const char s[] = "abcdefghijk";
    assert(std::regex_search(s, m, std::regex("cd((e)fg)hi")));

    std::match_results<const char*>::const_iterator i = m.begin();
    std::match_results<const char*>::const_iterator e = m.end();

    assert(static_cast<std::size_t>(e - i) == m.size());
    for (int j = 0; i != e; ++i, ++j)
        assert(*i == m[j]);
}

int main(int, char**)
{
    test();

  return 0;
}
