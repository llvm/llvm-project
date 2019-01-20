//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class multiset

// size_type size() const;

#include <set>
#include <cassert>

#include "min_allocator.h"

int main()
{
    {
    typedef std::multiset<int> M;
    M m;
    assert(m.size() == 0);
    m.insert(M::value_type(2));
    assert(m.size() == 1);
    m.insert(M::value_type(1));
    assert(m.size() == 2);
    m.insert(M::value_type(2));
    assert(m.size() == 3);
    m.erase(m.begin());
    assert(m.size() == 2);
    m.erase(m.begin());
    assert(m.size() == 1);
    m.erase(m.begin());
    assert(m.size() == 0);
    }
#if TEST_STD_VER >= 11
    {
    typedef std::multiset<int, std::less<int>, min_allocator<int>> M;
    M m;
    assert(m.size() == 0);
    m.insert(M::value_type(2));
    assert(m.size() == 1);
    m.insert(M::value_type(1));
    assert(m.size() == 2);
    m.insert(M::value_type(2));
    assert(m.size() == 3);
    m.erase(m.begin());
    assert(m.size() == 2);
    m.erase(m.begin());
    assert(m.size() == 1);
    m.erase(m.begin());
    assert(m.size() == 0);
    }
#endif
}
