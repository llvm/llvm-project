//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class match_results<BidirectionalIterator, Allocator>

// string_type str(size_type sub = 0) const;

#include <regex>
#include <cassert>
#include "test_macros.h"

void
test()
{
    std::match_results<const char*> m;
    const char s[] = "abcdefghijk";
    assert(std::regex_search(s, m, std::regex("cd((e)fg)hi")));
    assert(m.str() == std::string(m[0]));
    assert(m.str(0) == std::string(m[0]));
    assert(m.str(1) == std::string(m[1]));
    assert(m.str(2) == std::string(m[2]));
    assert(m.str(3) == std::string(m[3]));
    assert(m.str(4) == std::string(m[4]));
}

int main(int, char**)
{
    test();

  return 0;
}
