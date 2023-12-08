//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// class unordered_multiset

// allocator_type get_allocator() const

#include <unordered_set>
#include <cassert>

#include "test_allocator.h"
#include "test_macros.h"

int main(int, char**) {
    {
        std::allocator<int> alloc;
        const std::unordered_multiset<int> s(alloc);
        assert(s.get_allocator() == alloc);
    }
    {
        other_allocator<int> alloc(1);
        const std::unordered_multiset<int, std::hash<int>,
                                      std::equal_to<int>,
                                      other_allocator<int> > s(alloc);
        assert(s.get_allocator() == alloc);
    }

    return 0;
}
