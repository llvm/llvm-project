//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// void resize(size_type sz);

#include <vector>
#include <cassert>
#include "../../../stack_allocator.h"
#include "../../../MoveOnly.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        std::vector<MoveOnly> v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
    }
    {
        std::vector<MoveOnly, stack_allocator<MoveOnly, 300> > v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
    }
#else  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        std::vector<int> v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
    }
    {
        std::vector<int, stack_allocator<int, 300> > v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
