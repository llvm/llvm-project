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
#include "allocators.h"

int main(int, char**)
{
    {
        std::vector<int, test_allocator<int> > l(3, 2, test_allocator<int>(5));
        std::vector<int, test_allocator<int> > l2(l, test_allocator<int>(3));
        l2 = l;
        assert(l2 == l);
        assert(l2.get_allocator() == test_allocator<int>(3));
    }
    {
        std::vector<int, other_allocator<int> > l(3, 2, other_allocator<int>(5));
        std::vector<int, other_allocator<int> > l2(l, other_allocator<int>(3));
        l2 = l;
        assert(l2 == l);
        assert(l2.get_allocator() == other_allocator<int>(5));
    }
#if TEST_STD_VER >= 11
    {
        // Test with Allocator::propagate_on_container_copy_assignment == false_type
        using Alloc = NonPOCCAAllocator<int>;
        bool copy_assigned_into = false;
        std::vector<int, Alloc> l(3, 2, Alloc(5, nullptr));
        std::vector<int, Alloc> l2(l, Alloc(3, &copy_assigned_into));
        assert(!copy_assigned_into);
        l2 = l;
        assert(!copy_assigned_into);
        assert(l2 == l);
        assert(l2.get_allocator() == Alloc(3, nullptr));
    }
    {
        // Test with Allocator::propagate_on_container_copy_assignment == true_type
        // and equal allocators
        using Alloc = POCCAAllocator<int>;
        bool copy_assigned_into = false;
        std::vector<int, Alloc> l(3, 2, Alloc(5, nullptr));
        std::vector<int, Alloc> l2(l, Alloc(5, &copy_assigned_into));
        assert(!copy_assigned_into);
        l2 = l;
        assert(copy_assigned_into);
        assert(l2 == l);
        assert(l2.get_allocator() == Alloc(5, nullptr));
    }
    {
        // Test with Allocator::propagate_on_container_copy_assignment == true_type
        // and unequal allocators
        using Alloc = POCCAAllocator<int>;
        bool copy_assigned_into = false;
        std::vector<int, Alloc> l(3, 2, Alloc(5, nullptr));
        std::vector<int, Alloc> l2(l, Alloc(3, &copy_assigned_into));
        assert(!copy_assigned_into);
        l2 = l;
        assert(copy_assigned_into);
        assert(l2 == l);
        assert(l2.get_allocator() == Alloc(5, nullptr));
    }
    {
        std::vector<int, min_allocator<int> > l(3, 2, min_allocator<int>());
        std::vector<int, min_allocator<int> > l2(l, min_allocator<int>());
        l2 = l;
        assert(l2 == l);
        assert(l2.get_allocator() == min_allocator<int>());
    }
#endif

  return 0;
}
