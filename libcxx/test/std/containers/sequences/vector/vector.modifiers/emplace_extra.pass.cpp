//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <vector>

// template <class... Args> iterator emplace(const_iterator pos, Args&&... args);

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

int main(int, char**)
{
    {
        std::vector<int> v;
        v.reserve(3);
        assert(is_contiguous_container_asan_correct(v));
        v = { 1, 2, 3 };
        v.emplace(v.begin(), v.back());
        assert(v[0] == 3);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<int> v;
        v.reserve(4);
        assert(is_contiguous_container_asan_correct(v));
        v = { 1, 2, 3 };
        v.emplace(v.begin(), v.back());
        assert(v[0] == 3);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<int, min_allocator<int>> v;
        v.reserve(3);
        assert(is_contiguous_container_asan_correct(v));
        v = { 1, 2, 3 };
        v.emplace(v.begin(), v.back());
        assert(v[0] == 3);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<int, min_allocator<int>> v;
        v.reserve(4);
        assert(is_contiguous_container_asan_correct(v));
        v = { 1, 2, 3 };
        v.emplace(v.begin(), v.back());
        assert(v[0] == 3);
        assert(is_contiguous_container_asan_correct(v));
    }

  return 0;
}
