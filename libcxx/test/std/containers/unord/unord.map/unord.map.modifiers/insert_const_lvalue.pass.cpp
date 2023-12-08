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
// class unordered_map

// pair<iterator, bool> insert(const value_type& x);

#include <unordered_map>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"


template <class Container>
void do_insert_cv_test()
{
    typedef Container M;
    typedef std::pair<typename M::iterator, bool> R;
    typedef typename M::value_type VT;
    M m;

    const VT v1(2.5, 2);
    R r = m.insert(v1);
    assert(r.second);
    assert(m.size() == 1);
    assert(r.first->first == 2.5);
    assert(r.first->second == 2);

    const VT v2(2.5, 3);
    r = m.insert(v2);
    assert(!r.second);
    assert(m.size() == 1);
    assert(r.first->first == 2.5);
    assert(r.first->second == 2);

    const VT v3(1.5, 1);
    r = m.insert(v3);
    assert(r.second);
    assert(m.size() == 2);
    assert(r.first->first == 1.5);
    assert(r.first->second == 1);

    const VT v4(3.5, 3);
    r = m.insert(v4);
    assert(r.second);
    assert(m.size() == 3);
    assert(r.first->first == 3.5);
    assert(r.first->second == 3);

    const VT v5(3.5, 4);
    r = m.insert(v5);
    assert(!r.second);
    assert(m.size() == 3);
    assert(r.first->first == 3.5);
    assert(r.first->second == 3);
}

int main(int, char**)
{
    {
        typedef std::unordered_map<double, int> M;
        do_insert_cv_test<M>();
    }
#if TEST_STD_VER >= 11
    {
        typedef std::unordered_map<double, int, std::hash<double>, std::equal_to<double>,
                            min_allocator<std::pair<const double, int>>> M;
        do_insert_cv_test<M>();
    }
#endif

  return 0;
}
