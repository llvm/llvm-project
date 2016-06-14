//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// pointer data();

#include <vector>
#include <cassert>

#include "min_allocator.h"
#include "asan_testing.h"

int main()
{
    {
        std::vector<int> v;
        assert(v.data() == 0);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<int> v(100);
        assert(v.data() == &v.front());
        assert(is_contiguous_container_asan_correct(v));
    }
#if TEST_STD_VER >= 11
    {
        std::vector<int, min_allocator<int>> v;
        assert(v.data() == 0);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<int, min_allocator<int>> v(100);
        assert(v.data() == &v.front());
        assert(is_contiguous_container_asan_correct(v));
    }
#endif
}
