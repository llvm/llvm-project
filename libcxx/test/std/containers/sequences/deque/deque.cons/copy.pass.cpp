//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// deque(const deque&);

#include "asan_testing.h"
#include <deque>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class C>
void
test(const C& x)
{
    C c(x);
    assert(c == x);
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(x));
}

int main(int, char**)
{
    {
        int ab[] = {3, 4, 2, 8, 0, 1, 44, 34, 45, 96, 80, 1, 13, 31, 45};
        int* an = ab + sizeof(ab)/sizeof(ab[0]);
        test(std::deque<int>(ab, an));
    }
    {
        std::deque<int, test_allocator<int> > v(3, 2, test_allocator<int>(5));
        std::deque<int, test_allocator<int> > v2 = v;
        assert(v2 == v);
        assert(v2.get_allocator() == v.get_allocator());
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(v));
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(v2));
    }
#if TEST_STD_VER >= 11
    {
        std::deque<int, other_allocator<int> > v(3, 2, other_allocator<int>(5));
        std::deque<int, other_allocator<int> > v2 = v;
        assert(v2 == v);
        assert(v2.get_allocator() == other_allocator<int>(-2));
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(v));
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(v2));
    }
    {
        int ab[] = {3, 4, 2, 8, 0, 1, 44, 34, 45, 96, 80, 1, 13, 31, 45};
        int* an = ab + sizeof(ab)/sizeof(ab[0]);
        test(std::deque<int, min_allocator<int>>(ab, an));
    }
    {
        std::deque<int, min_allocator<int> > v(3, 2, min_allocator<int>());
        std::deque<int, min_allocator<int> > v2 = v;
        assert(v2 == v);
        assert(v2.get_allocator() == v.get_allocator());
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(v));
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(v2));
    }
#endif

  return 0;
}
