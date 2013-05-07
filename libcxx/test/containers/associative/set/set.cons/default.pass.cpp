//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// set();

#include <set>
#include <cassert>

int main()
{
    std::set<int> m;
    assert(m.empty());
    assert(m.begin() == m.end());
}
