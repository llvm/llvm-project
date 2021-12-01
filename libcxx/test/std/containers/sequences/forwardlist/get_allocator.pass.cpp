//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// class forward_list

// allocator_type get_allocator() const

#include <forward_list>
#include <cassert>

#include "test_allocator.h"
#include "test_macros.h"

int main(int, char**) {
    {
        std::allocator<int> alloc;
        const std::forward_list<int> fl(alloc);
        assert(fl.get_allocator() == alloc);
    }
    {
        other_allocator<int> alloc(1);
        const std::forward_list<int, other_allocator<int> > fl(alloc);
        assert(fl.get_allocator() == alloc);
    }

    return 0;
}
