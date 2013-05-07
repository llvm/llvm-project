//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <set>

// class multiset

// multiset();

#include <set>
#include <cassert>

int main()
{
    std::multiset<int> m;
    assert(m.empty());
    assert(m.begin() == m.end());
}
