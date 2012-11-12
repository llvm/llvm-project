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

// unordered_multiset(const unordered_multiset& u);

#include <unordered_set>
#include <cassert>
#include <cfloat>

#include "../../../test_compare.h"
#include "../../../test_hash.h"
#include "../../../test_allocator.h"

int main()
{
    {
        typedef std::unordered_multiset<int,
                                   test_hash<std::hash<int> >,
                                   test_compare<std::equal_to<int> >,
                                   test_allocator<int>
                                   > C;
        typedef int P;
        P a[] =
        {
            P(1),
            P(2),
            P(3),
            P(4),
            P(1),
            P(2)
        };
        C c0(a, a + sizeof(a)/sizeof(a[0]),
            7,
            test_hash<std::hash<int> >(8),
            test_compare<std::equal_to<int> >(9),
            test_allocator<int>(10)
           );
        C c = c0;
        assert(c.bucket_count() == 7);
        assert(c.size() == 6);
        C::const_iterator i = c.cbegin();
        assert(*i == 1);
        ++i;
        assert(*i == 1);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 3);
        ++i;
        assert(*i == 4);
        assert(c.hash_function() == test_hash<std::hash<int> >(8));
        assert(c.key_eq() == test_compare<std::equal_to<int> >(9));
        assert(c.get_allocator() == test_allocator<int>(10));
        assert(!c.empty());
        assert(std::distance(c.begin(), c.end()) == c.size());
        assert(std::distance(c.cbegin(), c.cend()) == c.size());
        assert(fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }
#ifndef _LIBCPP_HAS_NO_ADVANCED_SFINAE
    {
        typedef std::unordered_multiset<int,
                                   test_hash<std::hash<int> >,
                                   test_compare<std::equal_to<int> >,
                                   other_allocator<int>
                                   > C;
        typedef int P;
        P a[] =
        {
            P(1),
            P(2),
            P(3),
            P(4),
            P(1),
            P(2)
        };
        C c0(a, a + sizeof(a)/sizeof(a[0]),
            7,
            test_hash<std::hash<int> >(8),
            test_compare<std::equal_to<int> >(9),
            other_allocator<int>(10)
           );
        C c = c0;
        assert(c.bucket_count() == 7);
        assert(c.size() == 6);
        C::const_iterator i = c.cbegin();
        assert(*i == 1);
        ++i;
        assert(*i == 1);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 3);
        ++i;
        assert(*i == 4);
        assert(c.hash_function() == test_hash<std::hash<int> >(8));
        assert(c.key_eq() == test_compare<std::equal_to<int> >(9));
        assert(c.get_allocator() == other_allocator<int>(-2));
        assert(!c.empty());
        assert(std::distance(c.begin(), c.end()) == c.size());
        assert(std::distance(c.cbegin(), c.cend()) == c.size());
        assert(fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }
#endif  // _LIBCPP_HAS_NO_ADVANCED_SFINAE
}
