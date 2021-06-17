//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include <cmath>
#include <cstddef>

#include "test_macros.h"
#include "../../../check_consecutive.h"
#include "../../../test_compare.h"
#include "../../../test_hash.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef std::unordered_multiset<int,
                                   test_hash<std::hash<int> >,
                                   test_equal_to<int>,
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
            test_equal_to<int>(9),
            test_allocator<int>(10)
           );
        C c = c0;
        LIBCPP_ASSERT(c.bucket_count() == 7);
        assert(c.size() == 6);
        CheckConsecutiveValues<C::const_iterator>(c.find(1), c.end(), 1, 2);
        CheckConsecutiveValues<C::const_iterator>(c.find(2), c.end(), 2, 2);
        CheckConsecutiveValues<C::const_iterator>(c.find(3), c.end(), 3, 1);
        CheckConsecutiveValues<C::const_iterator>(c.find(4), c.end(), 4, 1);
        assert(c.hash_function() == test_hash<std::hash<int> >(8));
        assert(c.key_eq() == test_equal_to<int>(9));
        assert(c.get_allocator() == test_allocator<int>(10));
        assert(!c.empty());
        assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
        assert(static_cast<std::size_t>(std::distance(c.cbegin(), c.cend())) == c.size());
        assert(std::fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }
#if TEST_STD_VER >= 11
    {
        typedef std::unordered_multiset<int,
                                   test_hash<std::hash<int> >,
                                   test_equal_to<int>,
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
            test_equal_to<int>(9),
            other_allocator<int>(10)
           );
        C c = c0;
        LIBCPP_ASSERT(c.bucket_count() == 7);
        assert(c.size() == 6);
        CheckConsecutiveValues<C::const_iterator>(c.find(1), c.end(), 1, 2);
        CheckConsecutiveValues<C::const_iterator>(c.find(2), c.end(), 2, 2);
        CheckConsecutiveValues<C::const_iterator>(c.find(3), c.end(), 3, 1);
        CheckConsecutiveValues<C::const_iterator>(c.find(4), c.end(), 4, 1);
        assert(c.hash_function() == test_hash<std::hash<int> >(8));
        assert(c.key_eq() == test_equal_to<int>(9));
        assert(c.get_allocator() == other_allocator<int>(-2));
        assert(!c.empty());
        assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
        assert(static_cast<std::size_t>(std::distance(c.cbegin(), c.cend())) == c.size());
        assert(std::fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }
    {
        typedef std::unordered_multiset<int,
                                   test_hash<std::hash<int> >,
                                   test_equal_to<int>,
                                   min_allocator<int>
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
            test_equal_to<int>(9),
            min_allocator<int>()
           );
        C c = c0;
        LIBCPP_ASSERT(c.bucket_count() == 7);
        assert(c.size() == 6);
        CheckConsecutiveValues<C::const_iterator>(c.find(1), c.end(), 1, 2);
        CheckConsecutiveValues<C::const_iterator>(c.find(2), c.end(), 2, 2);
        CheckConsecutiveValues<C::const_iterator>(c.find(3), c.end(), 3, 1);
        CheckConsecutiveValues<C::const_iterator>(c.find(4), c.end(), 4, 1);
        assert(c.hash_function() == test_hash<std::hash<int> >(8));
        assert(c.key_eq() == test_equal_to<int>(9));
        assert(c.get_allocator() == min_allocator<int>());
        assert(!c.empty());
        assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
        assert(static_cast<std::size_t>(std::distance(c.cbegin(), c.cend())) == c.size());
        assert(std::fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }
#endif

  return 0;
}
