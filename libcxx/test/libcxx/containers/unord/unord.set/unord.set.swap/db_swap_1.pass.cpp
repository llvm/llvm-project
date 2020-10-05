//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_set

// void swap(unordered_set& x, unordered_set& y);

// This test requires debug mode, which the library on macOS doesn't have.
// UNSUPPORTED: with_system_cxx_lib=macosx

#define _LIBCPP_DEBUG 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <unordered_set>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        int a1[] = {1, 3, 7, 9, 10};
        int a2[] = {0, 2, 4, 5, 6, 8, 11};
        std::unordered_set<int> c1(a1, a1+sizeof(a1)/sizeof(a1[0]));
        std::unordered_set<int> c2(a2, a2+sizeof(a2)/sizeof(a2[0]));
        std::unordered_set<int>::iterator i1 = c1.begin();
        std::unordered_set<int>::iterator i2 = c2.begin();
        swap(c1, c2);
        c1.erase(i2);
        c2.erase(i1);
        std::unordered_set<int>::iterator j = i1;
        c1.erase(i1);
        assert(false);
    }

    return 0;
}
