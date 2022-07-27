//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// class vector<bool>

// allocator_type get_allocator() const

#include <vector>
#include <cassert>

#include "test_allocator.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool test() {
    {
        std::allocator<bool> alloc;
        const std::vector<bool> vb(alloc);
        assert(vb.get_allocator() == alloc);
    }
    {
        other_allocator<bool> alloc(1);
        const std::vector<bool, other_allocator<bool> > vb(alloc);
        assert(vb.get_allocator() == alloc);
    }

    return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
    static_assert(test());
#endif

    return 0;
}
