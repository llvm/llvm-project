//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// bool empty() const;

#include <set>
#include <cassert>

#include "min_allocator.h"

int main()
{
    {
    typedef std::set<int> M;
    M m;
    assert(m.empty());
    m.insert(M::value_type(1));
    assert(!m.empty());
    m.clear();
    assert(m.empty());
    }
#if TEST_STD_VER >= 11
    {
    typedef std::set<int, std::less<int>, min_allocator<int>> M;
    M m;
    assert(m.empty());
    m.insert(M::value_type(1));
    assert(!m.empty());
    m.clear();
    assert(m.empty());
    }
#endif
}
