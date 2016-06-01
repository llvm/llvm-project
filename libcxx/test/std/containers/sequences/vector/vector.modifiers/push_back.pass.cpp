//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// void push_back(const value_type& x);

#include <vector>
#include <cassert>
#include "../../../stack_allocator.h"
#include "min_allocator.h"
#include "asan_testing.h"

int main()
{
    {
        std::vector<int> c;
        c.push_back(0);
        assert(c.size() == 1);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(1);
        assert(c.size() == 2);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(2);
        assert(c.size() == 3);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(3);
        assert(c.size() == 4);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(4);
        assert(c.size() == 5);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
    }
    {
        std::vector<int, stack_allocator<int, 15> > c;
        c.push_back(0);
        assert(c.size() == 1);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(1);
        assert(c.size() == 2);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(2);
        assert(c.size() == 3);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(3);
        assert(c.size() == 4);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(4);
        assert(c.size() == 5);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
    }
#if __cplusplus >= 201103L
    {
        std::vector<int, min_allocator<int>> c;
        c.push_back(0);
        assert(c.size() == 1);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(1);
        assert(c.size() == 2);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(2);
        assert(c.size() == 3);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(3);
        assert(c.size() == 4);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(4);
        assert(c.size() == 5);
        assert(is_contiguous_container_asan_correct(c));
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
    }
#endif
}
