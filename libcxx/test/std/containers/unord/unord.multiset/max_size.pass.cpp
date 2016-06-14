//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_multiset

// size_type max_size() const;

#include <unordered_set>
#include <cassert>

#include "min_allocator.h"

int main()
{
    {
        std::unordered_multiset<int> u;
        assert(u.max_size() > 0);
    }
#if TEST_STD_VER >= 11
    {
        std::unordered_multiset<int, std::hash<int>,
                                      std::equal_to<int>, min_allocator<int>> u;
        assert(u.max_size() > 0);
    }
#endif
}
