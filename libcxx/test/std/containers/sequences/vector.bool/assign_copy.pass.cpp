//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// vector& operator=(const vector& c);

#include <vector>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX20 bool tests()
{
    {
        std::vector<bool, test_allocator<bool> > l(3, true, test_allocator<bool>(5));
        std::vector<bool, test_allocator<bool> > l2(l, test_allocator<bool>(3));
        l2 = l;
        assert(l2 == l);
        assert(l2.get_allocator() == test_allocator<bool>(3));
    }
    {
        std::vector<bool, other_allocator<bool> > l(3, true, other_allocator<bool>(5));
        std::vector<bool, other_allocator<bool> > l2(l, other_allocator<bool>(3));
        l2 = l;
        assert(l2 == l);
        assert(l2.get_allocator() == other_allocator<bool>(5));
    }
#if TEST_STD_VER >= 11
    {
        std::vector<bool, min_allocator<bool> > l(3, true, min_allocator<bool>());
        std::vector<bool, min_allocator<bool> > l2(l, min_allocator<bool>());
        l2 = l;
        assert(l2 == l);
        assert(l2.get_allocator() == min_allocator<bool>());
    }
#endif

    return true;
}

int main(int, char**)
{
    tests();
#if TEST_STD_VER > 17
    static_assert(tests());
#endif
    return 0;
}
