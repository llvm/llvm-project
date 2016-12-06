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

// unordered_multiset& operator=(initializer_list<value_type> il);

#include <unordered_set>
#include <cassert>
#include <cfloat>
#include <cstddef>

#include "../../../test_compare.h"
#include "../../../test_hash.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    {
        typedef test_allocator<int> A;
        typedef std::unordered_multiset<int,
                                   test_hash<std::hash<int> >,
                                   test_compare<std::equal_to<int> >,
                                   A
                                   > C;
        typedef int P;
        C c =   {
                    P(4),
                    P(1),
                    P(2)
                };
        c =     {
                    P(1),
                    P(2),
                    P(3),
                    P(4),
                    P(1),
                    P(2)
                };
        assert(c.bucket_count() >= 7);
        assert(c.size() == 6);
        assert(c.count(1) == 2);
        assert(c.count(2) == 2);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);
        assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
        assert(static_cast<std::size_t>(std::distance(c.cbegin(), c.cend())) == c.size());
        assert(fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }
#if TEST_STD_VER >= 11
    {
        typedef min_allocator<int> A;
        typedef std::unordered_multiset<int,
                                   test_hash<std::hash<int> >,
                                   test_compare<std::equal_to<int> >,
                                   A
                                   > C;
        typedef int P;
        C c =   {
                    P(4),
                    P(1),
                    P(2)
                };
        c =     {
                    P(1),
                    P(2),
                    P(3),
                    P(4),
                    P(1),
                    P(2)
                };
        assert(c.bucket_count() >= 7);
        assert(c.size() == 6);
        assert(c.count(1) == 2);
        assert(c.count(2) == 2);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);
        assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
        assert(static_cast<std::size_t>(std::distance(c.cbegin(), c.cend())) == c.size());
        assert(fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }
#endif
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
