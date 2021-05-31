//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_multimap

// unordered_multimap(const unordered_multimap& u);

#include <unordered_map>
#include <string>
#include <set>
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
        typedef std::unordered_multimap<int, std::string,
                                   test_hash<std::hash<int> >,
                                   test_equal_to<int>,
                                   test_allocator<std::pair<const int, std::string> >
                                   > C;
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
        C c0(a, a + sizeof(a)/sizeof(a[0]),
            7,
            test_hash<std::hash<int> >(8),
            test_equal_to<int>(9),
            test_allocator<std::pair<const int, std::string> >(10)
           );
        C c = c0;
        LIBCPP_ASSERT(c.bucket_count() == 7);
        assert(c.size() == 6);
        std::multiset<std::string> s;
        s.insert("one");
        s.insert("four");
        CheckConsecutiveKeys<C::const_iterator>(c.find(1), c.end(), 1, s);
        s.insert("two");
        s.insert("four");
        CheckConsecutiveKeys<C::const_iterator>(c.find(2), c.end(), 2, s);
        s.insert("three");
        CheckConsecutiveKeys<C::const_iterator>(c.find(3), c.end(), 3, s);
        s.insert("four");
        CheckConsecutiveKeys<C::const_iterator>(c.find(4), c.end(), 4, s);
        assert(c.hash_function() == test_hash<std::hash<int> >(8));
        assert(c.key_eq() == test_equal_to<int>(9));
        assert(c.get_allocator() ==
               (test_allocator<std::pair<const int, std::string> >(10)));
        assert(!c.empty());
        assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
        assert(static_cast<std::size_t>(std::distance(c.cbegin(), c.cend())) == c.size());
        assert(std::fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }
#if TEST_STD_VER >= 11
    {
        typedef std::unordered_multimap<int, std::string,
                                   test_hash<std::hash<int> >,
                                   test_equal_to<int>,
                                   other_allocator<std::pair<const int, std::string> >
                                   > C;
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
        C c0(a, a + sizeof(a)/sizeof(a[0]),
            7,
            test_hash<std::hash<int> >(8),
            test_equal_to<int>(9),
            other_allocator<std::pair<const int, std::string> >(10)
           );
        C c = c0;
        LIBCPP_ASSERT(c.bucket_count() == 7);
        assert(c.size() == 6);
        std::multiset<std::string> s;
        s.insert("one");
        s.insert("four");
        CheckConsecutiveKeys<C::const_iterator>(c.find(1), c.end(), 1, s);
        s.insert("two");
        s.insert("four");
        CheckConsecutiveKeys<C::const_iterator>(c.find(2), c.end(), 2, s);
        s.insert("three");
        CheckConsecutiveKeys<C::const_iterator>(c.find(3), c.end(), 3, s);
        s.insert("four");
        CheckConsecutiveKeys<C::const_iterator>(c.find(4), c.end(), 4, s);
        assert(c.hash_function() == test_hash<std::hash<int> >(8));
        assert(c.key_eq() == test_equal_to<int>(9));
        assert(c.get_allocator() ==
               (other_allocator<std::pair<const int, std::string> >(-2)));
        assert(!c.empty());
        assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
        assert(static_cast<std::size_t>(std::distance(c.cbegin(), c.cend())) == c.size());
        assert(std::fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }
    {
        typedef std::unordered_multimap<int, std::string,
                                   test_hash<std::hash<int> >,
                                   test_equal_to<int>,
                                   min_allocator<std::pair<const int, std::string> >
                                   > C;
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
        C c0(a, a + sizeof(a)/sizeof(a[0]),
            7,
            test_hash<std::hash<int> >(8),
            test_equal_to<int>(9),
            min_allocator<std::pair<const int, std::string> >()
           );
        C c = c0;
        LIBCPP_ASSERT(c.bucket_count() == 7);
        assert(c.size() == 6);
        std::multiset<std::string> s;
        s.insert("one");
        s.insert("four");
        CheckConsecutiveKeys<C::const_iterator>(c.find(1), c.end(), 1, s);
        s.insert("two");
        s.insert("four");
        CheckConsecutiveKeys<C::const_iterator>(c.find(2), c.end(), 2, s);
        s.insert("three");
        CheckConsecutiveKeys<C::const_iterator>(c.find(3), c.end(), 3, s);
        s.insert("four");
        CheckConsecutiveKeys<C::const_iterator>(c.find(4), c.end(), 4, s);
        assert(c.hash_function() == test_hash<std::hash<int> >(8));
        assert(c.key_eq() == test_equal_to<int>(9));
        assert(c.get_allocator() ==
               (min_allocator<std::pair<const int, std::string> >()));
        assert(!c.empty());
        assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
        assert(static_cast<std::size_t>(std::distance(c.cbegin(), c.cend())) == c.size());
        assert(std::fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }
#endif

  return 0;
}
