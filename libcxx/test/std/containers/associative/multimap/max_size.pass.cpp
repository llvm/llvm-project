//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <map>

// class multimap

// size_type max_size() const;

#include <map>
#include <cassert>

#include "min_allocator.h"

int main()
{
    {
    typedef std::multimap<int, double> M;
    M m;
    assert(m.max_size() != 0);
    }
#if TEST_STD_VER >= 11
    {
    typedef std::multimap<int, double, std::less<int>, min_allocator<std::pair<const int, double>>> M;
    M m;
    assert(m.max_size() != 0);
    }
#endif
}
