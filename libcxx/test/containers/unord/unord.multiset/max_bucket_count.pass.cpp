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

// size_type max_bucket_count() const;

#include <unordered_set>
#include <cassert>

int main()
{
    {
        typedef std::unordered_multiset<int> C;
        const C c;
        assert(c.max_bucket_count() > 0);
    }
}
