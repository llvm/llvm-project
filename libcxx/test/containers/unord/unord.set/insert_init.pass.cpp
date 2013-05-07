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
// class unordered_set

// void insert(initializer_list<value_type> il);

#include <unordered_set>
#include <cassert>

#include "test_iterators.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    {
        typedef std::unordered_set<int> C;
        typedef int P;
        C c;
        c.insert(
                    {
                        P(1),
                        P(2),
                        P(3),
                        P(4),
                        P(1),
                        P(2)
                    }
                );
        assert(c.size() == 4);
        assert(c.count(1) == 1);
        assert(c.count(2) == 1);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);
    }
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
