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
// class unordered_multimap

// size_type bucket_size(size_type n) const

#include <unordered_map>
#include <string>
#include <cassert>

int main()
{
    {
        typedef std::unordered_multimap<int, std::string> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(1, "one"),
            P(2, "two"),
            P(3, "three"),
            P(4, "four"),
            P(1, "four"),
            P(2, "four"),
        };
        const C c(std::begin(a), std::end(a));
        assert(c.bucket_count() >= 7);
        assert(c.bucket_size(0) == 0);
        assert(c.bucket_size(1) == 2);
        assert(c.bucket_size(2) == 2);
        assert(c.bucket_size(3) == 1);
        assert(c.bucket_size(4) == 1);
        assert(c.bucket_size(5) == 0);
        assert(c.bucket_size(6) == 0);
    }
}
