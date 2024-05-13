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
// class unordered_multimap

// unordered_multimap(size_type n, const allocator_type& alloc);

#include <unordered_map>
#include <cassert>
#include <iterator>

#include "test_macros.h"
#include "../../../NotConstructible.h"
#include "../../../test_compare.h"
#include "../../../test_hash.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class Allocator>
void test(const Allocator& alloc)
{
    typedef std::unordered_multimap<NotConstructible, NotConstructible,
                                    test_hash<NotConstructible>,
                                    test_equal_to<NotConstructible>,
                                    Allocator> C;
    C c(7,
        alloc);
    LIBCPP_ASSERT(c.bucket_count() == 7);
    assert(c.hash_function() == test_hash<NotConstructible>());
    assert(c.key_eq() == test_equal_to<NotConstructible>());
    assert(c.get_allocator() == alloc);
    assert(c.size() == 0);
    assert(c.empty());
    assert(std::distance(c.begin(), c.end()) == 0);
    assert(c.load_factor() == 0);
    assert(c.max_load_factor() == 1);
}

int main(int, char**)
{
    typedef std::pair<const NotConstructible, NotConstructible> V;
    test(test_allocator<V>(10));
    test(min_allocator<V>());
    test(explicit_allocator<V>());

    return 0;
}
