//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class multiset

// multiset();

#include <set>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    std::multiset<int> m;
    assert(m.empty());
    assert(m.begin() == m.end());
    }
#if TEST_STD_VER >= 11
    {
    std::multiset<int, std::less<int>, min_allocator<int>> m;
    assert(m.empty());
    assert(m.begin() == m.end());
    }
    {
    typedef explicit_allocator<int> A;
        {
        std::multiset<int, std::less<int>, A> m;
        assert(m.empty());
        assert(m.begin() == m.end());
        }
        {
        A a;
        std::multiset<int, std::less<int>, A> m(a);
        assert(m.empty());
        assert(m.begin() == m.end());
        }
    }
    {
    std::multiset<int> m = {};
    assert(m.empty());
    assert(m.begin() == m.end());
    }
#endif

  return 0;
}
