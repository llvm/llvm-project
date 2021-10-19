//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class multimap

// explicit multimap(const key_compare& comp);

// key_compare key_comp() const;

#include <map>
#include <cassert>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    typedef test_less<int> C;
    const std::multimap<int, double, C> m(C(3));
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.key_comp() == C(3));
    }
#if TEST_STD_VER >= 11
    {
    typedef test_less<int> C;
    const std::multimap<int, double, C, min_allocator<std::pair<const int, double>>> m(C(3));
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.key_comp() == C(3));
    }
#endif

  return 0;
}
