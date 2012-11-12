//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// void shrink_to_fit();

#include <vector>
#include <cassert>
#include "../../../stack_allocator.h"

int main()
{
    {
        std::vector<int> v(100);
        v.push_back(1);
        v.shrink_to_fit();
        assert(v.capacity() == 101);
        assert(v.size() == 101);
    }
    {
        std::vector<int, stack_allocator<int, 401> > v(100);
        v.push_back(1);
        v.shrink_to_fit();
        assert(v.capacity() == 101);
        assert(v.size() == 101);
    }
    {
        std::vector<int, stack_allocator<int, 400> > v(100);
        v.push_back(1);
        v.shrink_to_fit();
        assert(v.capacity() == 200);
        assert(v.size() == 101);
    }
}
