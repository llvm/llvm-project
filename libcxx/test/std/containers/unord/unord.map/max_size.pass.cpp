//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_map

// size_type max_size() const;

#include <unordered_map>
#include <cassert>

#include "min_allocator.h"

int main()
{
    {
        std::unordered_map<int, int> u;
        assert(u.max_size() > 0);
    }
#if TEST_STD_VER >= 11
    {
        std::unordered_map<int, int, std::hash<int>, std::equal_to<int>,
                                    min_allocator<std::pair<const int, int>>> u;
        assert(u.max_size() > 0);
    }
#endif
}
