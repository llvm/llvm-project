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

// size_type max_size() const;

#include <set>
#include <cassert>

#include "min_allocator.h"

int main()
{
    {
    typedef std::set<int> M;
    M m;
    assert(m.max_size() != 0);
    }
#if TEST_STD_VER >= 11
    {
    typedef std::set<int, std::less<int>, min_allocator<int>> M;
    M m;
    assert(m.max_size() != 0);
    }
#endif
}
