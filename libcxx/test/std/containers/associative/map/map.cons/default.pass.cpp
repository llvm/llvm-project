//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// map();

#include <map>
#include <cassert>

#include "min_allocator.h"

int main(int, char**)
{
    {
    std::map<int, double> m;
    assert(m.empty());
    assert(m.begin() == m.end());
    }
#if TEST_STD_VER >= 11
    {
    std::map<int, double, std::less<int>, min_allocator<std::pair<const int, double>>> m;
    assert(m.empty());
    assert(m.begin() == m.end());
    }
    {
    typedef explicit_allocator<std::pair<const int, double>> A;
        {
        std::map<int, double, std::less<int>, A> m;
        assert(m.empty());
        assert(m.begin() == m.end());
        }
        {
        A a;
        std::map<int, double, std::less<int>, A> m(a);
        assert(m.empty());
        assert(m.begin() == m.end());
        }
    }
    {
    std::map<int, double> m = {};
    assert(m.empty());
    assert(m.begin() == m.end());
    }
#endif

  return 0;
}
