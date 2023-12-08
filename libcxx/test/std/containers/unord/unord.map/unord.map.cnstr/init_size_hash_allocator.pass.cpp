//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_map

// unordered_map(initializer_list<value_type> il, size_type n,
//               const hasher& hash, const allocator_type& alloc);

#include <unordered_map>
#include <cassert>
#include <string>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <iterator>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "../../../test_hash.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class Allocator>
void test(const Allocator& alloc) {
    typedef std::unordered_map<int, std::string,
                               test_hash<int>,
                               test_equal_to<int>,
                               Allocator
                               > C;
    typedef std::pair<int, std::string> P;

    C c({
            P(1, "one"),
            P(2, "two"),
            P(3, "three"),
            P(4, "four"),
            P(1, "four"),
            P(2, "four"),
         },
         7,
         test_hash<int>(5),
         alloc);
    LIBCPP_ASSERT(c.bucket_count() == 7);
    assert(c.size() == 4);
    assert(c.at(1) == "one");
    assert(c.at(2) == "two");
    assert(c.at(3) == "three");
    assert(c.at(4) == "four");
    assert(c.hash_function() == test_hash<int>(5));
    assert(c.key_eq() == test_equal_to<int>());
    assert(c.get_allocator() == alloc);
    assert(!c.empty());
    assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
    assert(std::fabs(c.load_factor() - (float)c.size() / c.bucket_count()) < FLT_EPSILON);
    assert(c.max_load_factor() == 1);
}

int main(int, char**)
{
    typedef std::pair<const int, std::string> P;

    test(test_allocator<P>(10));
    test(min_allocator<P>());
    test(explicit_allocator<P>());

    return 0;
}
